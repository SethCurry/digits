#pragma once

#include "Leap.h"

#include <mutex>
#include <vector>

// Number of float features produced by modelInputs() per hand frame.
// Kept in sync with src/lib/libdigits/logging.cpp:
//   5 fingers * 4 joints * 3 coordinates + 3 palm + 3 wrist == 66.
inline constexpr int kInputsPerFrame = 66;

class LibdigitspyListener : public Leap::Listener {
public:
    void setBufferSize(int newSize);
    void setInterval(long newInterval);
    void getFrames(float** into);

    void onInit(const Leap::Controller&) override;
    void onConnect(const Leap::Controller&) override;
    void onDisconnect(const Leap::Controller&) override;
    void onExit(const Leap::Controller&) override;
    void onFrame(const Leap::Controller&) override;
    void onFocusGained(const Leap::Controller&) override;
    void onFocusLost(const Leap::Controller&) override;
    void onDeviceChange(const Leap::Controller&) override;
    void onServiceConnect(const Leap::Controller&) override;
    void onServiceDisconnect(const Leap::Controller&) override;

    // Set once `bufferedFrames` has filled up at least once; read by the
    // C API to know when a full window is available.
    bool full = false;

private:
    long lastEmit = 0;
    long interval = 50;
    int bufferSize = 50;
    std::vector<std::vector<float>> bufferedFrames;
    std::mutex bufferedFramesMut;
};
