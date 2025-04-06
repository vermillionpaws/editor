#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec4 fragColor;
layout(location = 1) in vec2 fragTexCoord;

layout(location = 0) out vec4 outColor;

layout(binding = 0) uniform sampler2D texSampler;

layout(push_constant) uniform PushConstants {
    mat4 transform;
    vec4 color;
    uint useTexture;
    uint _padding[3];
} pushConstants;

void main() {
    if (pushConstants.useTexture == 1) {
        vec4 texColor = texture(texSampler, fragTexCoord);
        outColor = fragColor * pushConstants.color * texColor;
    } else {
        outColor = fragColor * pushConstants.color;
    }
}
