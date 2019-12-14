#version 450


// uniforms
layout (set = 0, binding = 0) uniform texture2D tex;
layout (set = 0, binding = 1) uniform sampler samp;


// in
layout (location = 1) in vec2 frag_uv;

// out
layout (location = 0) out vec4 color;

void main(){
    color = texture(sampler2D(tex, samp), frag_uv);
}