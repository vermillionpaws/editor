#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec2 inPosition;
layout(location = 1) in vec4 inColor;
layout(location = 2) in vec2 inTexCoord;

layout(location = 0) out vec4 fragColor;
layout(location = 1) out vec2 fragTexCoord;

layout(push_constant) uniform PushConstants {
    mat4 transform;
    vec4 color;
    uint useTexture;
    uint _padding[3]; // Padding to match Rust struct
} pushConstants;

void main() {
    gl_Position = pushConstants.transform * vec4(inPosition, 0.0, 1.0);
    fragColor = inColor * pushConstants.color;
    fragTexCoord = inTexCoord;
}
