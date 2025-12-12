#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <switch.h>

#include <EGL/egl.h>
#include <glad/glad.h>

#define GLM_FORCE_PURE
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <glm/mat4x4.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "stb_image.h"
#include "cat_png.h"

constexpr auto TAU = glm::two_pi<float>();

#ifndef ENABLE_NXLINK
#define TRACE(fmt,...) ((void)0)
#else
#include <unistd.h>
#define TRACE(fmt,...) printf("%s: " fmt "\n", __PRETTY_FUNCTION__, ## __VA_ARGS__)
static int s_nxlinkSock = -1;
static void initNxLink() {
    if (R_FAILED(socketInitializeDefault())) return;
    s_nxlinkSock = nxlinkStdio();
    if (s_nxlinkSock >= 0) TRACE("printf output now goes to nxlink server");
    else socketExit();
}
static void deinitNxLink() {
    if (s_nxlinkSock >= 0) {
        close(s_nxlinkSock);
        socketExit();
        s_nxlinkSock = -1;
    }
}
extern "C" void userAppInit() { initNxLink(); }
extern "C" void userAppExit() { deinitNxLink(); }
#endif

static EGLDisplay s_display;
static EGLContext s_context;
static EGLSurface s_surface;

static bool initEgl(NWindow* win) {
    s_display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    if (!s_display) return false;
    eglInitialize(s_display, nullptr, nullptr);
    eglBindAPI(EGL_OPENGL_API);

    EGLConfig config;
    EGLint numConfigs;
    static const EGLint framebufferAttributeList[] = {
        EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
        EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8, EGL_ALPHA_SIZE, 8,
        EGL_DEPTH_SIZE, 24, EGL_STENCIL_SIZE, 8, EGL_NONE
    };
    eglChooseConfig(s_display, framebufferAttributeList, &config, 1, &numConfigs);
    if (numConfigs == 0) return false;

    s_surface = eglCreateWindowSurface(s_display, config, win, nullptr);
    if (!s_surface) return false;

    static const EGLint contextAttributeList[] = {
        EGL_CONTEXT_MAJOR_VERSION, 4,
        EGL_CONTEXT_MINOR_VERSION, 3,
        EGL_NONE
    };
    s_context = eglCreateContext(s_display, config, EGL_NO_CONTEXT, contextAttributeList);
    if (!s_context) return false;

    eglMakeCurrent(s_display, s_surface, s_surface, s_context);
    return true;
}

static void deinitEgl() {
    if (s_display) {
        eglMakeCurrent(s_display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
        if (s_context) eglDestroyContext(s_display, s_context);
        if (s_surface) eglDestroySurface(s_display, s_surface);
        eglTerminate(s_display);
    }
}

static const char* const vertexShaderSource = R"text(
#version 320 es
precision highp float;

layout(location=0) in vec3 inPos;
layout(location=1) in vec2 inTexCoord;
layout(location=2) in vec3 inNormal;
layout(location=3) in vec3 cubeOffset; // per-vertex cube offset (world position)

out vec2 vtxTexCoord;
out vec4 vtxNormalQuat;
out vec3 vtxView;
out vec3 vtxWorldPos;

uniform mat4 projMtx;
uniform float time;

void main() {
    // Compute per-vertex rotation based on cubeOffset (phase)
    float phase = cubeOffset.x + cubeOffset.y + cubeOffset.z;
    float angle = time + phase;
    vec3 axis = normalize(vec3(sin(phase), cos(phase*1.5), sin(phase*0.7)));

    float c = cos(angle);
    float s = sin(angle);
    mat4 rot = mat4(1.0);

    rot[0][0] = axis.x*axis.x*(1.0-c)+c;
    rot[0][1] = axis.x*axis.y*(1.0-c)-axis.z*s;
    rot[0][2] = axis.x*axis.z*(1.0-c)+axis.y*s;
    rot[1][0] = axis.y*axis.x*(1.0-c)+axis.z*s;
    rot[1][1] = axis.y*axis.y*(1.0-c)+c;
    rot[1][2] = axis.y*axis.z*(1.0-c)-axis.x*s;
    rot[2][0] = axis.z*axis.x*(1.0-c)-axis.y*s;
    rot[2][1] = axis.z*axis.y*(1.0-c)+axis.x*s;
    rot[2][2] = axis.z*axis.z*(1.0-c)+c;

    // Apply rotation to local position FIRST, then translate to world position
    vec3 rotatedPos = (rot * vec4(inPos, 1.0)).xyz;
    vec3 pos = rotatedPos + cubeOffset;

    vec4 worldPos = vec4(pos, 1.0);
    vtxWorldPos = worldPos.xyz;
    vtxView = -worldPos.xyz;

    vec3 normal = normalize(mat3(rot) * inNormal);
    float z = (1.0 + normal.z) / 2.0;
    vtxNormalQuat = vec4(1.0, 0.0, 0.0, 0.0);
    if (z > 0.0) { vtxNormalQuat.z = sqrt(z); vtxNormalQuat.xy = normal.xy / (2.0 * vtxNormalQuat.z); }

    vtxTexCoord = inTexCoord;
    gl_Position = projMtx * worldPos;
}
)text";

static const char* const fragmentShaderSource = R"text(
#version 320 es
precision highp float;

in vec2 vtxTexCoord;
in vec4 vtxNormalQuat;
in vec3 vtxView;
in vec3 vtxWorldPos;

out vec4 fragColor;

uniform vec4 lightPos;
uniform vec3 ambient;
uniform vec3 diffuse;
uniform vec4 specular;
uniform sampler2D tex_diffuse;

vec3 quatrotate(vec4 q, vec3 v){ return v + 2.0*cross(q.xyz, cross(q.xyz,v)+q.w*v); }

vec3 complexLighting(vec3 normal, vec3 lightVec, vec3 viewVec, vec3 texColor){
    vec3 result = ambient * texColor;
    // simple Blinn-ish specular using specular.w as shininess
    vec3 halfVec = normalize(viewVec + lightVec);
    result += diffuse * texColor * max(dot(normal, lightVec), 0.0);
    result += pow(max(dot(normal, halfVec), 0.0), specular.w) * specular.xyz;
    return result;
}

void main(){
    vec4 normquat = normalize(vtxNormalQuat);
    vec3 normal = quatrotate(normquat, vec3(0.0,0.0,1.0));
    vec3 lightVec = normalize(lightPos.xyz + vtxView);
    vec3 viewVec = normalize(vtxView);
    vec4 texColor = texture(tex_diffuse, vtxTexCoord);
    vec3 lightColor = complexLighting(normal, lightVec, viewVec, texColor.rgb);
    fragColor = vec4(min(lightColor, 1.0), texColor.a);
}
)text";

static GLuint createAndCompileShader(GLenum type, const char* source) {
    GLuint handle = glCreateShader(type);
    glShaderSource(handle,1,&source,nullptr);
    glCompileShader(handle);
    GLint success; glGetShaderiv(handle,GL_COMPILE_STATUS,&success);
    if(!success){ char buf[1024]; glGetShaderInfoLog(handle,sizeof(buf),nullptr,buf); TRACE("%u: %s", (unsigned)type, buf); glDeleteShader(handle); return 0; }
    return handle;
}

typedef struct {
    float position[3];
    float texcoord[2];
    float normal[3];
    float offset[3]; // cubeOffset (per-vertex, repeated per cube vertex)
} Vertex;

static const Vertex cubeVertices[] = {
    // 36 vertices (position, texcoord, normal). offset not used here - set later per-cube
    {{-0.5f,-0.5f, 0.5f},{0,0},{0,0,1},{0,0,0}},{{ 0.5f,-0.5f, 0.5f},{1,0},{0,0,1},{0,0,0}},{{ 0.5f, 0.5f, 0.5f},{1,1},{0,0,1},{0,0,0}},
    {{ 0.5f, 0.5f, 0.5f},{1,1},{0,0,1},{0,0,0}},{{-0.5f, 0.5f, 0.5f},{0,1},{0,0,1},{0,0,0}},{{-0.5f,-0.5f, 0.5f},{0,0},{0,0,1},{0,0,0}},

    {{-0.5f,-0.5f,-0.5f},{0,0},{0,0,-1},{0,0,0}},{{-0.5f, 0.5f,-0.5f},{1,0},{0,0,-1},{0,0,0}},{{ 0.5f, 0.5f,-0.5f},{1,1},{0,0,-1},{0,0,0}},
    {{ 0.5f, 0.5f,-0.5f},{1,1},{0,0,-1},{0,0,0}},{{ 0.5f,-0.5f,-0.5f},{0,1},{0,0,-1},{0,0,0}},{{-0.5f,-0.5f,-0.5f},{0,0},{0,0,-1},{0,0,0}},

    {{ 0.5f,-0.5f,-0.5f},{0,0},{1,0,0},{0,0,0}},{{ 0.5f, 0.5f,-0.5f},{1,0},{1,0,0},{0,0,0}},{{ 0.5f, 0.5f, 0.5f},{1,1},{1,0,0},{0,0,0}},
    {{ 0.5f, 0.5f, 0.5f},{1,1},{1,0,0},{0,0,0}},{{ 0.5f,-0.5f, 0.5f},{0,1},{1,0,0},{0,0,0}},{{ 0.5f,-0.5f,-0.5f},{0,0},{1,0,0},{0,0,0}},

    {{-0.5f,-0.5f,-0.5f},{0,0},{-1,0,0},{0,0,0}},{{-0.5f,-0.5f, 0.5f},{1,0},{-1,0,0},{0,0,0}},{{-0.5f, 0.5f, 0.5f},{1,1},{-1,0,0},{0,0,0}},
    {{-0.5f, 0.5f, 0.5f},{1,1},{-1,0,0},{0,0,0}},{{-0.5f, 0.5f,-0.5f},{0,1},{-1,0,0},{0,0,0}},{{-0.5f,-0.5f,-0.5f},{0,0},{-1,0,0},{0,0,0}},

    {{-0.5f, 0.5f,-0.5f},{0,0},{0,1,0},{0,0,0}},{{-0.5f, 0.5f, 0.5f},{1,0},{0,1,0},{0,0,0}},{{ 0.5f, 0.5f, 0.5f},{1,1},{0,1,0},{0,0,0}},
    {{ 0.5f, 0.5f, 0.5f},{1,1},{0,1,0},{0,0,0}},{{ 0.5f, 0.5f,-0.5f},{0,1},{0,1,0},{0,0,0}},{{-0.5f, 0.5f,-0.5f},{0,0},{0,1,0},{0,0,0}},

    {{-0.5f,-0.5f,-0.5f},{0,0},{0,-1,0},{0,0,0}},{{ 0.5f,-0.5f,-0.5f},{1,0},{0,-1,0},{0,0,0}},{{ 0.5f,-0.5f, 0.5f},{1,1},{0,-1,0},{0,0,0}},
    {{ 0.5f,-0.5f, 0.5f},{1,1},{0,-1,0},{0,0,0}},{{-0.5f,-0.5f, 0.5f},{0,1},{0,-1,0},{0,0,0}},{{-0.5f,-0.5f,-0.5f},{0,0},{0,-1,0},{0,0,0}},
};
static const int cubeVertexCount = sizeof(cubeVertices)/sizeof(Vertex);

#define GRID_SIZE 96
#define CUBE_SPACING 2.5f

static GLuint s_program = 0;
static GLuint s_vao = 0;
static GLuint s_vbo = 0;
static GLuint s_tex = 0;
static GLint loc_projMtx = -1, loc_lightPos = -1, loc_ambient = -1, loc_diffuse = -1, loc_specular = -1, loc_tex_diffuse = -1, loc_time = -1;
static u64 s_startTicks = 0;
static GLsizei s_totalVertexCount = 0;

static bool allocateChecked(void* p){
    if(!p){ TRACE("allocation failed"); return false; }
    return true;
}

static void sceneInit() {
    GLuint vsh = createAndCompileShader(GL_VERTEX_SHADER, vertexShaderSource);
    GLuint fsh = createAndCompileShader(GL_FRAGMENT_SHADER, fragmentShaderSource);
    s_program = glCreateProgram();
    glAttachShader(s_program, vsh);
    glAttachShader(s_program, fsh);
    glLinkProgram(s_program);
    GLint linkok = GL_FALSE;
    glGetProgramiv(s_program, GL_LINK_STATUS, &linkok);
    if(!linkok){
        char buf[1024]; glGetProgramInfoLog(s_program, sizeof(buf), nullptr, buf);
        TRACE("program link error: %s", buf);
    }
    glDeleteShader(vsh); glDeleteShader(fsh);

    loc_projMtx = glGetUniformLocation(s_program,"projMtx");
    loc_lightPos = glGetUniformLocation(s_program,"lightPos");
    loc_ambient = glGetUniformLocation(s_program,"ambient");
    loc_diffuse = glGetUniformLocation(s_program,"diffuse");
    loc_specular = glGetUniformLocation(s_program,"specular");
    loc_tex_diffuse = glGetUniformLocation(s_program,"tex_diffuse");
    loc_time = glGetUniformLocation(s_program,"time");

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    glEnable(GL_CULL_FACE);

    // --- Prepare VAO/VBO ---
    glGenVertexArrays(1, &s_vao);
    glGenBuffers(1, &s_vbo);
    glBindVertexArray(s_vao);
    glBindBuffer(GL_ARRAY_BUFFER, s_vbo);

    // Build full vertex buffer with offsets as a separate attribute (not baked into positions)
    const int totalCubes = GRID_SIZE * GRID_SIZE * GRID_SIZE;
    s_totalVertexCount = cubeVertexCount * totalCubes;
    TRACE("Allocating vertex buffer: cubes=%d vertices=%d", totalCubes, s_totalVertexCount);

    // allocate huge buffer
    size_t vbSize = sizeof(Vertex) * (size_t)s_totalVertexCount;
    Vertex* vertexBuffer = (Vertex*)malloc(vbSize);
    if(!allocateChecked(vertexBuffer)) abort();
    Vertex* ptr = vertexBuffer;

    for(int x = 0; x < GRID_SIZE; x++){
        for(int y = 0; y < GRID_SIZE; y++){
            for(int z = 0; z < GRID_SIZE; z++){
                // cube center offsets (world coordinates)
                glm::vec3 offset = glm::vec3(
                    (x - GRID_SIZE/2.0f) * CUBE_SPACING,
                    (y - GRID_SIZE/2.0f) * CUBE_SPACING,
                    (z - GRID_SIZE/2.0f) * CUBE_SPACING - 40.0f
                );

                for(int i = 0; i < cubeVertexCount; ++i){
                    // copy base vertex but DO NOT bake offset into position; store offset in attribute
                    ptr->position[0] = cubeVertices[i].position[0];
                    ptr->position[1] = cubeVertices[i].position[1];
                    ptr->position[2] = cubeVertices[i].position[2];
                    ptr->texcoord[0] = cubeVertices[i].texcoord[0];
                    ptr->texcoord[1] = cubeVertices[i].texcoord[1];
                    ptr->normal[0] = cubeVertices[i].normal[0];
                    ptr->normal[1] = cubeVertices[i].normal[1];
                    ptr->normal[2] = cubeVertices[i].normal[2];
                    ptr->offset[0] = offset.x;
                    ptr->offset[1] = offset.y;
                    ptr->offset[2] = offset.z;
                    ptr++;
                }
            }
        }
    }

    // upload big VBO
    glBufferData(GL_ARRAY_BUFFER, vbSize, vertexBuffer, GL_STATIC_DRAW);
    // free CPU copy (we keep no per-instance data)
    free(vertexBuffer);

    // vertex attributes: 0 position, 1 texcoord, 2 normal, 3 offset
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, position));
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, texcoord));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, normal));
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, offset));
    glEnableVertexAttribArray(3);

    glBindVertexArray(0);

    // --- Load texture ---
    glGenTextures(1, &s_tex);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, s_tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    int width, height, nchan;
    stbi_uc* img = stbi_load_from_memory((const stbi_uc*)cat_png, cat_png_size, &width, &height, &nchan, 4);
    if(img){
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, img);
        stbi_image_free(img);
    } else {
        TRACE("failed to load embedded texture");
    }

    glUseProgram(s_program);
    auto projMtx = glm::perspective(40.0f * TAU / 360.0f, 1280.0f / 720.0f, 0.01f, 1000.0f);
    glUniformMatrix4fv(loc_projMtx, 1, GL_FALSE, glm::value_ptr(projMtx));
    glUniform4f(loc_lightPos, 2.0f, 2.0f, 3.0f, 1.0f);
    glUniform3f(loc_ambient, 0.10f, 0.10f, 0.10f);
    glUniform3f(loc_diffuse, 0.7f, 0.7f, 0.7f);
    glUniform4f(loc_specular, 0.8f, 0.8f, 0.8f, 32.0f);
    glUniform1i(loc_tex_diffuse, 0);

    s_startTicks = armGetSystemTick();
}

static float getTime(){ return ((armGetSystemTick()-s_startTicks)*625.0f/12.0f)/1000000000.0f; }

static void sceneRender(){
    glClearColor(0x25/255.0f, 0x25/255.0f, 0x35/255.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glUseProgram(s_program);
    // update time uniform
    glUniform1f(loc_time, getTime());
    glBindVertexArray(s_vao);
    // Draw the full prebaked buffer (single draw call)
    glDrawArrays(GL_TRIANGLES, 0, s_totalVertexCount);
    glBindVertexArray(0);
}

static void sceneExit(){
    if(s_tex) { glDeleteTextures(1, &s_tex); s_tex = 0; }
    if(s_vbo) { glDeleteBuffers(1, &s_vbo); s_vbo = 0; }
    if(s_vao) { glDeleteVertexArrays(1, &s_vao); s_vao = 0; }
    if(s_program){ glDeleteProgram(s_program); s_program = 0; }
}

int main(int argc,char* argv[]){
    if(!initEgl(nwindowGetDefault())) { TRACE("egl init failed"); return EXIT_FAILURE; }
    gladLoadGL();
    sceneInit();
    padConfigureInput(1, HidNpadStyleSet_NpadStandard);
    PadState pad;
    padInitializeDefault(&pad);

    while(appletMainLoop()){
        padUpdate(&pad);
        if(padGetButtonsDown(&pad) & HidNpadButton_Plus) break;
        sceneRender();
        eglSwapInterval(s_display, 0);
        eglSwapBuffers(s_display, s_surface);
    }

    sceneExit();
    deinitEgl();
    return EXIT_SUCCESS;
}
