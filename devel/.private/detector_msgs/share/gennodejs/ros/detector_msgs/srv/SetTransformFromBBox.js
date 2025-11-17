// Auto-generated. Do not edit!

// (in-package detector_msgs.srv)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;
let BBox = require('../msg/BBox.js');

//-----------------------------------------------------------


//-----------------------------------------------------------

class SetTransformFromBBoxRequest {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.bbox = null;
      this.frame = null;
    }
    else {
      if (initObj.hasOwnProperty('bbox')) {
        this.bbox = initObj.bbox
      }
      else {
        this.bbox = new BBox();
      }
      if (initObj.hasOwnProperty('frame')) {
        this.frame = initObj.frame
      }
      else {
        this.frame = '';
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type SetTransformFromBBoxRequest
    // Serialize message field [bbox]
    bufferOffset = BBox.serialize(obj.bbox, buffer, bufferOffset);
    // Serialize message field [frame]
    bufferOffset = _serializer.string(obj.frame, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type SetTransformFromBBoxRequest
    let len;
    let data = new SetTransformFromBBoxRequest(null);
    // Deserialize message field [bbox]
    data.bbox = BBox.deserialize(buffer, bufferOffset);
    // Deserialize message field [frame]
    data.frame = _deserializer.string(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += BBox.getMessageSize(object.bbox);
    length += _getByteLength(object.frame);
    return length + 4;
  }

  static datatype() {
    // Returns string type for a service object
    return 'detector_msgs/SetTransformFromBBoxRequest';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return 'e54e737b8466a302e35424be325fe2f4';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    detector_msgs/BBox bbox
    string frame
    
    ================================================================================
    MSG: detector_msgs/BBox
    string label
    float64 score
    int64 x
    int64 y
    int64 w
    int64 h
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new SetTransformFromBBoxRequest(null);
    if (msg.bbox !== undefined) {
      resolved.bbox = BBox.Resolve(msg.bbox)
    }
    else {
      resolved.bbox = new BBox()
    }

    if (msg.frame !== undefined) {
      resolved.frame = msg.frame;
    }
    else {
      resolved.frame = ''
    }

    return resolved;
    }
};

class SetTransformFromBBoxResponse {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.result = null;
    }
    else {
      if (initObj.hasOwnProperty('result')) {
        this.result = initObj.result
      }
      else {
        this.result = false;
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type SetTransformFromBBoxResponse
    // Serialize message field [result]
    bufferOffset = _serializer.bool(obj.result, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type SetTransformFromBBoxResponse
    let len;
    let data = new SetTransformFromBBoxResponse(null);
    // Deserialize message field [result]
    data.result = _deserializer.bool(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    return 1;
  }

  static datatype() {
    // Returns string type for a service object
    return 'detector_msgs/SetTransformFromBBoxResponse';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return 'eb13ac1f1354ccecb7941ee8fa2192e8';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    bool result
    
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new SetTransformFromBBoxResponse(null);
    if (msg.result !== undefined) {
      resolved.result = msg.result;
    }
    else {
      resolved.result = false
    }

    return resolved;
    }
};

module.exports = {
  Request: SetTransformFromBBoxRequest,
  Response: SetTransformFromBBoxResponse,
  md5sum() { return 'cf10ef4f67b53145111605419d2c8caa'; },
  datatype() { return 'detector_msgs/SetTransformFromBBox'; }
};
