// Copyright 2019 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Framework exposed headers

#import <CoreVideo/CoreVideo.h>
#import <Foundation/Foundation.h>

@interface Landmark : NSObject
- (instancetype)init;
@property float x;
@property float y;
@property float z;
@property float visibility;
@property float presence;
@end

@protocol PoseResultDelegate <NSObject>
@optional
- (void)didReceive:(NSArray<Landmark *> *) landmarks;
@end

@interface PoseTracking : NSObject
- (instancetype)init;
- (void)startGraph;
- (void)processVideoFrame:(CVPixelBufferRef)imageBuffer;
@property(weak, nonatomic) id<PoseResultDelegate> delegate;
@property(nonatomic) size_t timestamp;
@end