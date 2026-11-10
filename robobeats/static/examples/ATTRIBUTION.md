# Example clip

`perception_test_video_29.mp4` is video `video_29` from the [Perception Test](https://github.com/google-deepmind/perception_test)
benchmark (Pătrăucean et al., NeurIPS 2023, Google DeepMind), licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Changes: audio track and container metadata removed; the
video stream is copied unchanged, so the demo server and the offline command (`python -m robobeats`) read the same frames.

The conditions and truth spans in `examples.json` are the benchmark's own action-localisation classes and labelled
spans for this video, word for word (the class names follow the Something-Something style, hence "something").
The video is not in the RoboBeats training data.
