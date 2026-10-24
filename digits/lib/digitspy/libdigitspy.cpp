#include "./include/libdigitspy.hpp"

#include "libdigits.hpp"

#include <chrono>
#include <iostream>
#include <thread>

namespace sc = std::chrono;

namespace {
LibdigitspyListener listener;
Leap::Controller controller;
}  // namespace

extern "C" {

// `duration` is accepted for ABI compatibility with the Python binding but is
// currently unused; the buffer size is fixed at the listener's default.
void start_listener(int interval, int duration) {
    (void)duration;
    listener.setInterval(interval);
    controller.addListener(listener);
}

void get_frames(float** dest) {
    while (!listener.full) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    listener.getFrames(dest);
}

}  // extern "C"

void LibdigitspyListener::setInterval(long newInterval) {
    interval = newInterval;
}

void LibdigitspyListener::setBufferSize(int newBufferSize) {
    bufferSize = newBufferSize;
}

void LibdigitspyListener::onInit(const Leap::Controller&) {
    std::cerr << "Initialized\n";
}

void LibdigitspyListener::onConnect(const Leap::Controller&) {
    std::cerr << "Connected\n";
}

void LibdigitspyListener::onDisconnect(const Leap::Controller&) {
    // Note: not dispatched when running in a debugger.
    std::cerr << "Disconnected\n";
}

void LibdigitspyListener::onExit(const Leap::Controller&) {
    std::cerr << "Exited\n";
}

void LibdigitspyListener::onFrame(const Leap::Controller& controller) {
    const auto now = sc::duration_cast<sc::milliseconds>(
                         sc::system_clock::now().time_since_epoch())
                         .count();

    if (now < lastEmit + interval) {
        return;
    }

    const Leap::Frame frame = controller.frame();
    const Leap::HandList hands = frame.hands();
    if (hands.count() != 1) {
        return;
    }

    for (const Leap::Hand& hand : hands) {
        std::vector<float> inputs = modelInputs(hand);

        std::lock_guard<std::mutex> guard(bufferedFramesMut);
        bufferedFrames.insert(bufferedFrames.begin(), std::move(inputs));

        while (static_cast<int>(bufferedFrames.size()) > bufferSize) {
            full = true;
            bufferedFrames.pop_back();
        }
    }

    lastEmit = now;
}

void LibdigitspyListener::getFrames(float** into) {
    std::lock_guard<std::mutex> guard(bufferedFramesMut);
    for (int i = 0; i < bufferSize; ++i) {
        const auto& frame = bufferedFrames.at(static_cast<std::size_t>(i));
        for (int j = 0; j < kInputsPerFrame; ++j) {
            into[i][j] = frame.at(static_cast<std::size_t>(j));
        }
    }
}

void LibdigitspyListener::onFocusGained(const Leap::Controller&) {
    std::cerr << "Focus Gained\n";
}

void LibdigitspyListener::onFocusLost(const Leap::Controller&) {
    std::cerr << "Focus Lost\n";
}

void LibdigitspyListener::onDeviceChange(const Leap::Controller& controller) {
    std::cerr << "Device Changed\n";
    const Leap::DeviceList devices = controller.devices();
    for (const Leap::Device& device : devices) {
        std::cerr << "id: " << device.toString() << "\n";
        std::cerr << "  isStreaming: " << (device.isStreaming() ? "true" : "false") << "\n";
    }
}

void LibdigitspyListener::onServiceConnect(const Leap::Controller&) {
    std::cerr << "Service Connected\n";
}

void LibdigitspyListener::onServiceDisconnect(const Leap::Controller&) {
    std::cerr << "Service Disconnected\n";
}
