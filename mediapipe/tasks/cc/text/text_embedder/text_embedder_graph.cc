/* Copyright 2022 The MediaPipe Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mediapipe/calculators/tensor/inference_calculator.pb.h"
#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/api2/port.h"
#include "mediapipe/framework/calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/tasks/cc/components/containers/proto/embeddings.pb.h"
#include "mediapipe/tasks/cc/components/processors/embedding_postprocessing_graph.h"
#include "mediapipe/tasks/cc/components/processors/proto/embedding_postprocessing_graph_options.pb.h"
#include "mediapipe/tasks/cc/components/proto/text_preprocessing_graph_options.pb.h"
#include "mediapipe/tasks/cc/components/text_preprocessing_graph.h"
#include "mediapipe/tasks/cc/core/model_resources.h"
#include "mediapipe/tasks/cc/core/model_task_graph.h"
#include "mediapipe/tasks/cc/core/proto/model_resources_calculator.pb.h"
#include "mediapipe/tasks/cc/text/text_embedder/proto/text_embedder_graph_options.pb.h"

namespace mediapipe::tasks::text::text_embedder {
namespace {

using ::mediapipe::api2::Input;
using ::mediapipe::api2::Output;
using ::mediapipe::api2::builder::Graph;
using ::mediapipe::api2::builder::Source;
using ::mediapipe::tasks::components::containers::proto::EmbeddingResult;
using ::mediapipe::tasks::core::ModelResources;

constexpr char kEmbeddingsTag[] = "EMBEDDINGS";
constexpr char kTextTag[] = "TEXT";
constexpr char kMetadataExtractorTag[] = "METADATA_EXTRACTOR";
constexpr char kTensorsTag[] = "TENSORS";

}  // namespace

// A "mediapipe.tasks.text.TextEmbedderGraph" performs text embedding
// extraction.
// - Accepts input text and outputs embeddings on CPU.
//
// Inputs:
//   TEXT - std::string
//     Input text to perform embedding extraction on.
//
// Outputs:
//   EMBEDDINGS - EmbeddingResult
//     The embedding result.
//
// Example:
// node {
//   calculator: "mediapipe.tasks.text.TextEmbedderGraph"
//   input_stream: "TEXT:text_in"
//   output_stream: "EMBEDDINGS:embedding_result_out"
//   options {
//     [mediapipe.tasks.text.text_embedder.proto.TextEmbedderGraphOptions.ext] {
//       base_options {
//         model_asset {
//           file_name: "/path/to/model.tflite"
//         }
//       }
//     }
//   }
// }
class TextEmbedderGraph : public core::ModelTaskGraph {
 public:
  absl::StatusOr<CalculatorGraphConfig> GetConfig(
      SubgraphContext* sc) override {
    CHECK(sc != nullptr);
    ASSIGN_OR_RETURN(const ModelResources* model_resources,
                     CreateModelResources<proto::TextEmbedderGraphOptions>(sc));
    Graph graph;
    ASSIGN_OR_RETURN(
        Source<EmbeddingResult> embedding_result_out,
        BuildTextEmbedderTask(sc->Options<proto::TextEmbedderGraphOptions>(),
                              *model_resources,
                              graph[Input<std::string>(kTextTag)], graph));
    embedding_result_out >> graph[Output<EmbeddingResult>(kEmbeddingsTag)];
    return graph.GetConfig();
  }

 private:
  // Adds a mediapipe TextEmbedder task graph into the provided
  // builder::Graph instance. The TextEmbedder task takes an input
  // text (std::string) and returns an embedding result.
  //
  // task_options: the mediapipe tasks TextEmbedderGraphOptions proto.
  // model_resources: the ModelResources object initialized from a
  //   TextEmbedder model file with model metadata.
  // text_in: (std::string) stream to run embedding extraction on.
  // graph: the mediapipe builder::Graph instance to be updated.
  absl::StatusOr<Source<EmbeddingResult>> BuildTextEmbedderTask(
      const proto::TextEmbedderGraphOptions& task_options,
      const ModelResources& model_resources, Source<std::string> text_in,
      Graph& graph) {
    // Adds preprocessing calculators and connects them to the text input
    // stream.
    auto& preprocessing =
        graph.AddNode("mediapipe.tasks.components.TextPreprocessingSubgraph");
    MP_RETURN_IF_ERROR(components::ConfigureTextPreprocessingSubgraph(
        model_resources,
        preprocessing.GetOptions<
            tasks::components::proto::TextPreprocessingGraphOptions>()));
    text_in >> preprocessing.In(kTextTag);

    // Adds both InferenceCalculator and ModelResourcesCalculator.
    auto& inference = AddInference(
        model_resources, task_options.base_options().acceleration(), graph);
    // The metadata extractor side-output comes from the
    // ModelResourcesCalculator.
    inference.SideOut(kMetadataExtractorTag) >>
        preprocessing.SideIn(kMetadataExtractorTag);
    preprocessing.Out(kTensorsTag) >> inference.In(kTensorsTag);

    // Adds postprocessing calculators and connects its input stream to the
    // inference results.
    auto& postprocessing = graph.AddNode(
        "mediapipe.tasks.components.processors.EmbeddingPostprocessingGraph");
    MP_RETURN_IF_ERROR(components::processors::ConfigureEmbeddingPostprocessing(
        model_resources, task_options.embedder_options(),
        &postprocessing.GetOptions<components::processors::proto::
                                       EmbeddingPostprocessingGraphOptions>()));
    inference.Out(kTensorsTag) >> postprocessing.In(kTensorsTag);

    // Outputs the embedding result.
    return postprocessing[Output<EmbeddingResult>(kEmbeddingsTag)];
  }
};

REGISTER_MEDIAPIPE_GRAPH(
    ::mediapipe::tasks::text::text_embedder::TextEmbedderGraph);

}  // namespace mediapipe::tasks::text::text_embedder
