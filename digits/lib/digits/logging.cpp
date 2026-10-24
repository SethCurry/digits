#include "libdigits.hpp"

#include "json.hpp"

#include <array>
#include <fstream>
#include <iostream>
#include <istream>
#include <vector>

using namespace Leap;
using json = nlohmann::json;

static std::vector<float> vecToVec(const Vector& vector) {
    return {vector.x, vector.y, vector.z};
}

static Vector avgVectors(const Vector& one, const Vector& two) {
    return Vector((one.x + two.x) / 2.0f,
                  (one.y + two.y) / 2.0f,
                  (one.z + two.z) / 2.0f);
}

// Builds the flat list of inputs used by the inference model, in the same
// order as SkeletalHGRFrame.to_frame() in digipy/training.py.
//
// For each finger in [thumb, index, middle, ring, pinky] we emit the
// first_knuckle, second_knuckle, third_knuckle and tip positions (each as
// x, y, z), followed by the palm and wrist positions. The knuckle positions
// are the midpoints between adjacent bone joints, matching the Python code.
std::vector<float> modelInputs(Hand hand) {
    std::vector<float> dim;
    dim.reserve(66);

    // Index fingers by type so we emit them in a fixed order regardless of
    // the order Leap reports them in.
    std::array<Finger, 5> fingersByType{};
    const FingerList fingers = hand.fingers();
    for (const Finger& finger : fingers) {
        const int type = static_cast<int>(finger.type());
        if (type >= 0 && type < 5) {
            fingersByType[static_cast<std::size_t>(type)] = finger;
        }
    }

    for (const Finger& finger : fingersByType) {
        const Bone metacarpal = finger.bone(Bone::TYPE_METACARPAL);
        const Bone proximal = finger.bone(Bone::TYPE_PROXIMAL);
        const Bone middle = finger.bone(Bone::TYPE_INTERMEDIATE);
        const Bone distal = finger.bone(Bone::TYPE_DISTAL);

        const Vector firstKnuckle =
            avgVectors(proximal.prevJoint(), metacarpal.nextJoint());
        const Vector secondKnuckle =
            avgVectors(middle.prevJoint(), proximal.nextJoint());
        const Vector thirdKnuckle =
            avgVectors(distal.prevJoint(), middle.nextJoint());
        const Vector tip = distal.nextJoint();

        for (const Vector& joint : {firstKnuckle, secondKnuckle, thirdKnuckle, tip}) {
            dim.push_back(joint.x);
            dim.push_back(joint.y);
            dim.push_back(joint.z);
        }
    }

    const Vector palm = hand.palmPosition();
    dim.push_back(palm.x);
    dim.push_back(palm.y);
    dim.push_back(palm.z);

    const Vector wrist = hand.arm().wristPosition();
    dim.push_back(wrist.x);
    dim.push_back(wrist.y);
    dim.push_back(wrist.z);

    return dim;
}

static json boneToJson(const Bone& bone, Bone::Type boneType) {
    json jBone;
    jBone["name"] = boneNames[boneType];
    jBone["start"] = vecToVec(bone.prevJoint());
    jBone["end"] = vecToVec(bone.nextJoint());
    jBone["direction"] = vecToVec(bone.direction());
    return jBone;
}

static json fingerToJson(const Finger& finger) {
    json jFinger;
    jFinger["name"] = fingerNames[finger.type()];
    jFinger["id"] = finger.id();
    jFinger["length"] = finger.length();
    jFinger["width"] = finger.width();

    std::vector<json> bonesVec;
    bonesVec.reserve(4);
    for (int b = 0; b < 4; ++b) {
        const auto boneType = static_cast<Bone::Type>(b);
        bonesVec.push_back(boneToJson(finger.bone(boneType), boneType));
    }
    jFinger["bones"] = bonesVec;
    return jFinger;
}

static json handToJson(const Hand& hand) {
    json j;
    j["is_left"] = hand.isLeft();
    j["id"] = hand.id();

    const Vector palmPos = hand.palmPosition();
    j["palm_position"] = std::vector<float>{palmPos.x, palmPos.y, palmPos.z};

    const Vector normal = hand.palmNormal();
    const Vector direction = hand.direction();
    j["pitch"] = direction.pitch();
    j["roll"] = normal.roll();
    j["yaw"] = direction.yaw();

    const Arm arm = hand.arm();
    json jArm;
    const Vector armDirection = arm.direction();
    jArm["direction"] =
        std::vector<float>{armDirection.x, armDirection.y, armDirection.z};
    jArm["wrist_position"] = vecToVec(arm.wristPosition());
    jArm["elbow_position"] = vecToVec(arm.elbowPosition());
    j["arm"] = jArm;

    std::vector<json> fingersVec;
    const FingerList fingers = hand.fingers();
    for (const Finger& finger : fingers) {
        fingersVec.push_back(fingerToJson(finger));
    }
    j["fingers"] = fingersVec;
    return j;
}

ModelLogger::ModelLogger(std::string outPath) {
    if (outPath == "stdout") {
        output = &std::cout;
    } else {
        file.open(outPath);
        output = &file;
        if (!file.is_open()) {
            std::cerr << "Failed to open output file " << outPath
                      << "; logging to stdout instead" << std::endl;
            output = &std::cout;
        }
    }
}

void ModelLogger::logFrame(Frame frame) const {
    if (frame.hands().count() < 1) {
        return;
    }

    json j;
    j["frame_id"] = frame.id();
    j["timestamp"] = frame.timestamp();
    j["num_hands"] = frame.hands().count();
    j["num_extended_fingers"] = frame.fingers().extended().count();
    j["num_gestures"] = frame.gestures().count();
    j["num_tools"] = frame.tools().count();

    std::vector<json> handsVec;
    const HandList hands = frame.hands();
    for (const Hand& hand : hands) {
        handsVec.push_back(handToJson(hand));
    }
    j["hands"] = handsVec;

    *output << j.dump() << std::endl;
}

void ModelLogger::logModelFrame(Frame frame) const {
    if (frame.hands().count() < 1) {
        return;
    }

    // Emit one comma-separated line of model inputs per hand.
    const HandList hands = frame.hands();

    std::ostream& out = *output;
    for (const Hand& hand : hands) {
        const std::vector<float> inputs = modelInputs(hand);

        bool first = true;
        for (const float value : inputs) {
            if (!first) {
                out << ",";
            }
            out << value;
            first = false;
        }
        out << std::endl;
    }
}
