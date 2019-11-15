#version 450
// push constants
layout (push_constant) uniform PushConsts {
  float time;
  float aspect_ratio;
} push;

// uniforms
layout (set = 0, binding = 0) uniform texture2D tex;
layout (set = 0, binding = 1) uniform sampler samp;


// in
layout (location = 1) in vec3 frag_color;
layout (location = 2) in vec2 frag_uv;

// out
layout (location = 0) out vec4 color;

vec3 rotate(vec3 p, float angle, vec3 axis)
{
    vec3 a = normalize(axis);
    float s = sin(angle);
    float c = cos(angle);
    float r = 1.0 - c;
    mat3 m = mat3(
        a.x * a.x * r + c,
        a.y * a.x * r + a.z * s,
        a.z * a.x * r - a.y * s,
        a.x * a.y * r - a.z * s,
        a.y * a.y * r + c,
        a.z * a.y * r + a.x * s,
        a.x * a.z * r + a.y * s,
        a.y * a.z * r - a.x * s,
        a.z * a.z * r + c
    );
    return m * p;
}

float map(vec3 p){
  vec3 w = p;
  float m = dot(w,w);

  vec4 trap = vec4(abs(w),m);
	float dz = 1.;
    
    
	for( int i=0; i<4; i++ )
  {
    float power = 3. + 5. * abs(cos(push.time / 4.));
    dz = power*pow(sqrt(m),power-1.)*dz + 1.0;
		// dz = 8.0*pow(m,3.5)*dz + 1.0;
    
    float r = length(w);
    float b = power*acos( w.y/r) + push.time * 2.;
    float a = power*atan( w.x, w.z ) - push.time * 3.;
    w = p + pow(r,power) * vec3( sin(b)*sin(a), cos(b), sin(b)*cos(a) );
    trap = min( trap, vec4(abs(w),m) );

    m = dot(w,w);
		if( m > 256.0 )
            break;
  }

  return 0.25*log(m)*sqrt(m)/dz;
}

void main(){
    // float time01 = -0.9 * abs(sin(push.time * 0.7)) + 0.9;
    // vec4 tex_color = texture(sampler2D(tex, samp), frag_uv);

    vec2 uv = vec2(frag_uv.x, 1. - frag_uv.y);

    uv = (uv*2.) - 1.;
    uv.x *= push.aspect_ratio;
    vec4 c = vec4(uv, 0.75, 1.0);

    vec3 ro = vec3(0. , 0., cos(0)) * 2.;
    vec3 rd = vec3(uv, (1.-dot(uv,uv)));
    rd = rotate(rd, 3.14159265, vec3(1., 0., 0.));

    // t is total distance
    // d is step distance
    float i = 0., t = 0., d = 0.;
    vec3 p;
    // raymarching loop
    for(i = 0.; i < 100.; i++){
      p = ro + rd * t;
      d = map(p);
      t += d*.75;
      if(d < 0.001 || d > 100.){break;}
    }

    c = 1. - vec4(i / 100.);

    // color = mix(tex_color, c, time01);
    color = c;
}