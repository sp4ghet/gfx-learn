#version 450
// push constants
layout (push_constant) uniform PushConsts {
  float time;
} push;

// uniforms
layout (set = 0, binding = 0) uniform texture2D tex;
layout (set = 0, binding = 1) uniform sampler samp;


// in
layout (location = 1) in vec3 frag_color;
layout (location = 2) in vec2 frag_uv;

// out
layout (location = 0) out vec4 color;


void main(){
    float time01 = -0.9 * abs(sin(push.time * 0.7)) + 0.9;
    vec4 tex_color = texture(sampler2D(tex, samp), frag_uv);
    vec2 uv = vec2(frag_uv.x, 1. - frag_uv.y);
    vec4 c = vec4(uv, 0.75, 1.0);
    color = mix(tex_color, c, time01);
}