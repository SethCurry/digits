#include "./include/libdigits.hpp"

#include <chrono>
#include <fstream>
#include <iostream>

using namespace Leap;

namespace sc = std::chrono;

LibdigitsListener::LibdigitsListener(std::string outputPath, int format, long interval)
    : format(format), interval(interval), output(outputPath) {}


void LibdigitsListener::onInit(const Controller&) {
    std::cerr << "Initialized" << std::endl;
}

void LibdigitsListener::onConnect(const Controller& controller) {
    std::cerr << "Connected" << std::endl;
    controller.enableGesture(Gesture::TYPE_CIRCLE);
    controller.enableGesture(Gesture::TYPE_KEY_TAP);
    controller.enableGesture(Gesture::TYPE_SCREEN_TAP);
    controller.enableGesture(Gesture::TYPE_SWIPE);
}

void LibdigitsListener::onDisconnect(const Controller&) {
    // Note: not dispatched when running in a debugger.
    std::cerr << "Disconnected" << std::endl;
}

void LibdigitsListener::onExit(const Controller&) {
    std::cerr << "Exited" << std::endl;
}

void LibdigitsListener::onFrame(const Controller& controller) {
    const auto now = sc::duration_cast<sc::milliseconds>(
                         sc::system_clock::now().time_since_epoch()).count();

    if (now >= lastEmit + interval) {
        const Frame frame = controller.frame();

        if (format == FORMAT_FULL) {
                output.logFrame(frame);
        } else if (format == FORMAT_MODEL) {
                output.logModelFrame(frame);
        }

        lastEmit = now;
    }
}

void LibdigitsListener::onFocusGained(const Controller&) {
    std::cerr << "Focus Gained" << std::endl;
}

void LibdigitsListener::onFocusLost(const Controller&) {
    std::cerr << "Focus Lost" << std::endl;
}

void LibdigitsListener::onDeviceChange(const Controller& controller) {
    std::cout << "Device Changed" << std::endl;
    const DeviceList devices = controller.devices();

    for (int i = 0; i < devices.count(); ++i) {
        std::cerr << "id: " << devices[i].toString() << std::endl;
        std::cerr << "  isStreaming: "
                  << (devices[i].isStreaming() ? "true" : "false") << std::endl;
    }
}

void LibdigitsListener::onServiceConnect(const Controller&) {
    std::cerr << "Service Connected" << std::endl;
}

void LibdigitsListener::onServiceDisconnect(const Controller&) {
    std::cerr << "Service Disconnected" << std::endl;
}
