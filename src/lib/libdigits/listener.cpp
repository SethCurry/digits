
#include <chrono>
#include <iostream>
#include "Leap.h"
#include "./include/libdigits.hpp"

using namespace Leap;

namespace sc = std::chrono;

void LibdigitsListener::setInterval(long newInterval) {
    interval = newInterval;
}

void LibdigitsListener::setFormat(int newFormat) {
    format = newFormat;
}

void LibdigitsListener::onInit(const Controller& controller) {
  std::cerr << "Initialized" << std::endl;
}

void LibdigitsListener::onConnect(const Controller& controller) {
  std::cerr << "Connected" << std::endl;
  controller.enableGesture(Gesture::TYPE_CIRCLE);
  controller.enableGesture(Gesture::TYPE_KEY_TAP);
  controller.enableGesture(Gesture::TYPE_SCREEN_TAP);
  controller.enableGesture(Gesture::TYPE_SWIPE);
}

void LibdigitsListener::onDisconnect(const Controller& controller) {
  // Note: not dispatched when running in a debugger.
  std::cerr << "Disconnected" << std::endl;
}

void LibdigitsListener::onExit(const Controller& controller) {
  std::cerr << "Exited" << std::endl;
}

void LibdigitsListener::onFrame(const Controller &controller) {
    auto time = sc::system_clock::now(); // get the current time
    auto since_epoch = time.time_since_epoch(); // get the duration since epoch

    // I don't know what system_clock returns
    // I think it's uint64_t nanoseconds since epoch
    // Either way this duration_cast will do the right thing
    auto millis = sc::duration_cast<sc::milliseconds>(since_epoch);

    long now = millis.count(); // just like java (new Date()).getTime();

    if (now >= lastEmit + interval) {
        const Frame frame = controller.frame();

        if (format == FORMAT_FULL) {
            logFrame(frame);
        }

        if (format == FORMAT_MODEL) {
            logModelFrame(frame);
        }

       lastEmit = now;
    }
}

void LibdigitsListener::onFocusGained(const Controller& controller) {
  std::cerr << "Focus Gained" << std::endl;
}

void LibdigitsListener::onFocusLost(const Controller& controller) {
  std::cerr << "Focus Lost" << std::endl;
}

void LibdigitsListener::onDeviceChange(const Controller& controller) {
  std::cout << "Device Changed" << std::endl;
  const DeviceList devices = controller.devices();

  for (int i = 0; i < devices.count(); ++i) {
    std::cerr << "id: " << devices[i].toString() << std::endl;
    std::cerr << "  isStreaming: " << (devices[i].isStreaming() ? "true" : "false") << std::endl;
  }
}

void LibdigitsListener::onServiceConnect(const Controller& controller) {
  std::cerr << "Service Connected" << std::endl;
}

void LibdigitsListener::onServiceDisconnect(const Controller& controller) {
  std::cerr << "Service Disconnected" << std::endl;
}
