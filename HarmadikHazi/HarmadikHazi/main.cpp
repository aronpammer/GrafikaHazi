//=============================================================================================
// Szamitogepes grafika hazi feladat keret. Ervenyes 2016-tol.
// A //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// sorokon beluli reszben celszeru garazdalkodni, mert a tobbit ugyis toroljuk.
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kivéve
// - new operatort hivni a lefoglalt adat korrekt felszabaditasa nelkul
// - felesleges programsorokat a beadott programban hagyni
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Pammer Áron
// Neptun : 
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha    barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================

#define _USE_MATH_DEFINES
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#if defined(__APPLE__)
#include <GLUT/GLUT.h>
#include <OpenGL/gl3.h>
#include <OpenGL/glu.h>
#else
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__)
#include <windows.h>
#endif
#include <GL/glew.h>
#include <GL/freeglut.h>
#endif

const unsigned int windowWidth = 600, windowHeight = 600;

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Innentol modosithatod...

#define PERIODIC false

int majorVersion = 3, minorVersion = 0;

void getErrorInfo(unsigned int handle) {
    int logLen;
    glGetShaderiv(handle, GL_INFO_LOG_LENGTH, &logLen);
    if (logLen > 0) {
        char * log = new char[logLen];
        int written;
        glGetShaderInfoLog(handle, logLen, &written, log);
        printf("Shader log:\n%s", log);
        delete log;
    }
}

void checkShader(unsigned int shader, char * message) {
    int OK;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &OK);
    if (!OK) {
        printf("%s!\n", message);
        getErrorInfo(shader);
    }
}

void checkLinking(unsigned int program) {
    int OK;
    glGetProgramiv(program, GL_LINK_STATUS, &OK);
    if (!OK) {
        printf("Failed to link shader program!\n");
        getErrorInfo(program);
    }
}

const char *vertexSource = R"(
#version 130
precision highp float;

uniform mat4 MVP, M, Minv;
uniform vec4 YwLiPos;
uniform vec4 CwLiPos;
uniform vec4 wEye;

in vec3 vtxPos;
in vec3 vtxNorm;

in vec2 uv;
out vec2 uvFragment;
out vec3 wNormal;
out vec3 wView;
out vec3 YwLight;
out vec3 CwLight;

void main() {
    gl_Position = vec4(vtxPos, 1) * MVP;
    vec4 wPos = vec4(vtxPos, 1) * M;
    YwLight  = YwLiPos.xyz * wPos.w - wPos.xyz * YwLiPos.w;
    CwLight  = CwLiPos.xyz * wPos.w - wPos.xyz * CwLiPos.w;
    wView   = wEye.xyz * wPos.w - wPos.xyz;
    wNormal = (Minv * vec4(vtxNorm, 0)).xyz;
    uvFragment = uv;
}
)";

const char *fragmentSource = R"(
#version 130
precision highp float;

uniform vec4 ks, ka;
uniform vec4 YLa, YLe;
uniform vec4 CLa, CLe;
uniform float shine;
uniform float darab;
uniform sampler2D samplerUnit;
in vec2 uvFragment;

in  vec3 wNormal;
in  vec3 wView;
in  vec3 YwLight;
in  vec3 CwLight;
out vec4 fragmentColor;

void main() {
    float s = uvFragment.x;
    float t = uvFragment.y;
    
    int sum = int(s*darab) + int(t*darab);
    vec4 diffuseColor = texture(samplerUnit, vec2(mod(sum, 2.0), 0));
    
    vec3 N = normalize(wNormal);
    vec3 V = normalize(wView);
    vec3 YL = normalize(YwLight);
    vec3 CL = normalize(CwLight);
    vec3 YH = normalize(YL + V);
    vec3 CH = normalize(YL + V);
    float intensity = 5;
    float Ycost = max(dot(N,YL), 0), Ycosd = max(dot(N,YH), 0);
    vec3 color = (ka.xyz * YLa.xyz + (diffuseColor.xyz * Ycost + ks.xyz * pow(Ycosd,shine)) * YLe.xyz) * intensity/(length(YwLight) * length(YwLight));
    float Ccost = max(dot(N,CL), 0), Ccosd = max(dot(N,CH), 0);
    vec3 color2 = (ka.xyz * CLa.xyz + (diffuseColor.xyz * Ccost + ks.xyz * pow(Ccosd,shine)) * CLe.xyz) * intensity/(length(CwLight) * length(CwLight));
    fragmentColor = vec4(color + color2,1);
}


)";

struct mat4 {
    float m[4][4];
public:
    mat4() {}
    mat4(float m00, float m01, float m02, float m03,
         float m10, float m11, float m12, float m13,
         float m20, float m21, float m22, float m23,
         float m30, float m31, float m32, float m33) {
        m[0][0] = m00; m[0][1] = m01; m[0][2] = m02; m[0][3] = m03;
        m[1][0] = m10; m[1][1] = m11; m[1][2] = m12; m[1][3] = m13;
        m[2][0] = m20; m[2][1] = m21; m[2][2] = m22; m[2][3] = m23;
        m[3][0] = m30; m[3][1] = m31; m[3][2] = m32; m[3][3] = m33;
    }
    
    mat4 operator*(const mat4& right) {
        mat4 result;
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++) {
                result.m[i][j] = 0;
                for (int k = 0; k < 4; k++)
                    result.m[i][j] += m[i][k] * right.m[k][j];
            }
        return result;
    }
    
    void SetUniform(unsigned int shader, const char* name) {
        int location = glGetUniformLocation(shader, name);
        if (location >= 0) glUniformMatrix4fv(location, 1, GL_TRUE, &m[0][0]); // set uniform variable MVP to the MVPTransform
        else printf("Error %s", name);
    }
};

mat4 Translate(float tx, float ty, float tz) {
    return mat4(
                1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1, 0,
                tx, ty, tz, 1);
}
mat4 Rotate(float theta, float wx, float wy, float wz)
{
    return mat4(
                1-(wx*wx-1)*(cos(theta)-1),             -wz*sin(theta)-wx*wy*(cos(theta)-1),    wy*sin(theta)-wx*wz*(cos(theta)-1),     0,
                wz*sin(theta)-wx*wy*(cos(theta)-1),     1-(wy*wy-1)*(cos(theta)-1),             -wx*sin(theta)-wy*wz*(cos(theta)-1),    0,
                -wy*sin(theta)-wx*wz*(cos(theta)-1),    wx*sin(theta)-wy*wz*(cos(theta)-1),     1-(wz*wz-1)*(cos(theta)-1),             0,
                0,                                      0,                                      0,                                      1
                );
}

mat4 Scale(float sx, float sy, float sz)
{
    return mat4(
                sx,     0,      0,      0,
                0,      sy,     0,      0,
                0,      0,      sz,     0,
                0,      0,      0,      1);
}


struct vec4 {
    union
    {
        float vertex[4];
        struct
        {
            float x, y, z, w;
        };
    };
    
    vec4(float x = 0, float y = 0, float z = 0, float w = 1) : x(x), y(y), z(z), w(w) {}
    vec4 operator+(const vec4& vec) {
        vec4 result;
        for (int i = 0; i < 3; i++)
            result.vertex[i] = this->vertex[i] + vec.vertex[i];
        return result;
    }
    vec4 operator*(const mat4& mat) {
        vec4 result;
        for (int j = 0; j < 4; j++) {
            result.vertex[j] = 0;
            for (int i = 0; i < 4; i++) result.vertex[j] += vertex[i] * mat.m[i][j];
        }
        return result;
    }
    vec4 operator*(const float& f) {
        vec4 result;
        for (int i = 0; i < 3; i++)
            result.vertex[i] = this->vertex[i] * f;
        return result;
    }
    vec4& operator=(const float& f) {
        for(int i = 0; i < 3; i++)
            this->vertex[i] = f;
        this->w = 1;
        return (*this);
    }
    vec4 operator-(const vec4& vec) {
        vec4 result;
        for (int i = 0; i < 3; i++)
            result.vertex[i] = this->vertex[i] - vec.vertex[i];
        return result;
    }
    float length()
    {
        return sqrtf(x*x + y*y + z*z);
    }
    vec4 normalize()
    {
        vec4 normalized = vec4(x, y, z, 1) * (1/length());
        return normalized;
    }
    friend float dot(const vec4 left, const vec4 right)
    {
        float result = 0;
        
        for (int i = 0; i < 3; ++i) {
            result += left.vertex[i] * right.vertex[i];
        }
        return result;
    }
    friend vec4 cross(const vec4 left, const vec4 right) {
        
        float x = left.y*right.z - left.z*right.y;
        float y = left.z*right.x - left.x*right.z;
        float z = left.x*right.y - left.y*right.x;
        return vec4(x,y,z);
    }
    friend vec4 reflect(vec4 inDir, vec4 normal)
    {
        vec4 result = inDir - (normal * (dot(normal, inDir) * 2));
        return result;
    }
    void SetUniform(unsigned int shader, const char* name) {
        int location = glGetUniformLocation(shader, name);
        if (location >= 0) glUniform4fv(location, 1, vertex);
        else printf("Error %s", name);
    }
};

struct Shader {
    unsigned int shaderProgram;
    
    void Create(const char * vsSrc, const char * vsAttrNames[], int numOfAttr,
                const char * fsSrc, const char * fsOuputName) {
        unsigned int vs = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(vs, 1, &vsSrc, NULL); glCompileShader(vs);
        unsigned int fs = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(fs, 1, &fsSrc, NULL); glCompileShader(fs);
        shaderProgram = glCreateProgram();
        glAttachShader(shaderProgram, vs);
        glAttachShader(shaderProgram, fs);
        
        for (int i = 0; i < numOfAttr; i++)
        {
            glBindAttribLocation(shaderProgram, i, vsAttrNames[i]);
        }
        glBindFragDataLocation(shaderProgram, 0, fsOuputName);
        glLinkProgram(shaderProgram);
    }
    void Bind() { glUseProgram(shaderProgram); }
} Phong;


struct Camera {
    vec4 wEye, wLookat, wVup, forward, up, right;
    float fov, asp, fp, bp;
public:
    
    Camera(vec4 wEye, vec4 wLookat, vec4 wVup, float fov, float fp, float bp) :
    wEye(wEye),
    wLookat(wLookat),
    wVup(wVup),
    fov(fov),
    fp(fp),
    bp(bp),
    asp(windowWidth / windowHeight) { }
    
    void Create() {
        wEye.SetUniform(Phong.shaderProgram, "wEye");
    }
    
    mat4 ViewMatrix() {
        return
        mat4(
             1, 0, 0, 0,
             0, 1, 0, 0,
             0, 0, 1, 0,
             -wEye.x, -wEye.y, -wEye.z, 1
             ) *
        mat4(up.x, right.x, forward.x, 0.0f,
             up.y, right.y, forward.y, 0.0f,
             up.z, right.z, forward.z, 0.0f,
             0.0f, 0.0f, 0.0f, 1.0
             );
    }
    mat4 V()
    {
        forward = (wEye - wLookat).normalize();
        up = cross(wVup, forward).normalize();
        right = cross(forward, up).normalize();
        return ViewMatrix();
    }
    mat4 P() {
        float sy = 1/tan(fov/2.0f);
        return mat4(sy/asp,     0.0f,       0.0f,                   0.0f,
                    0.0f,       sy,         0.0f,                   0.0f,
                    0.0f,       0.0f,       -(fp+bp) / (bp-fp),     -1.0f,
                    0.0f,       0.0f,       -2*fp*bp / (bp-fp),     0.0f);
    }
    
    void Animate(float t) {
    }
} camera = Camera(vec4(6.0f, 0.0f, 0.f),
                  vec4(6.8f, 0.0f, 0.0f)*Rotate(-M_PI/2.5f, 0, 1, 0), //rotate it to the left
                  vec4(0.0f, 1.0f, 0.f), M_PI/3,
                  0.01f,
                  50.0f);
struct Texture {
    unsigned int textureId;
    Texture() {
    }
    void Create()
    {
        glGenTextures(1, &textureId);
        glBindTexture(GL_TEXTURE_2D, textureId);    // binding
        int width = 2, height = 1;
        float image[6];
        image[0] = 0;
        image[1] = 0;
        image[2] = 1;
        image[3] = 1;
        image[4] = 0;
        image[5] = 0;
        
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height,
                     0, GL_RGB, GL_FLOAT, image); //Texture -> OpenGL
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        
    }
};
Texture texture;
struct Geometry {
    unsigned int vao, nVtx;
    
    Geometry( ) {
    }
    void Draw() {
        int samplerUnit = GL_TEXTURE0;
        int location = glGetUniformLocation(Phong.shaderProgram, "samplerUnit");
        glUniform1i(location, 0);
        if(location == -1)
            printf("failed to attach samplerunit\n");
        else
        {
            glActiveTexture(samplerUnit);
            glBindTexture(GL_TEXTURE_2D, texture.textureId);
        }
        
        glBindVertexArray(vao); glDrawArrays(GL_TRIANGLES, 0, nVtx);
    }
};

struct VertexData {
    vec4 position, normal;
    float u, v;
};

struct ParamSurface : Geometry {
    virtual VertexData GenVertexData(float u, float v) = 0;
    void Create(int N, int M)
    {
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);
        
        nVtx = N * M * 6;
        unsigned int vbo;
        glGenBuffers(1, &vbo); glBindBuffer(GL_ARRAY_BUFFER, vbo);
        
        VertexData *vtxData = new VertexData[nVtx];
        VertexData* pVtx = vtxData;
        for (int i = 0; i < N; i++) for (int j = 0; j < M; j++) {
            *pVtx++ = GenVertexData(static_cast<float>(i) / N, static_cast<float>(j) / M);
            *pVtx++ = GenVertexData(static_cast<float>(i + 1) / N, static_cast<float>(j) / M);
            *pVtx++ = GenVertexData(static_cast<float>(i) / N, static_cast<float>(j + 1) / M);
            *pVtx++ = GenVertexData(static_cast<float>(i + 1) / N, static_cast<float>(j) / M);
            *pVtx++ = GenVertexData(static_cast<float>(i + 1) / N, static_cast<float>(j + 1) / M);
            *pVtx++ = GenVertexData(static_cast<float>(i) / N, static_cast<float>(j + 1) / M);
        }
        int stride = sizeof(VertexData), sVec3 = sizeof(vec4);
        glBufferData(GL_ARRAY_BUFFER, nVtx * stride, vtxData, GL_STATIC_DRAW);
        
        glEnableVertexAttribArray(0);
        glEnableVertexAttribArray(1);
        glEnableVertexAttribArray(2);
        glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,stride,(void*)0);
        glVertexAttribPointer(1,3,GL_FLOAT,GL_FALSE,stride,(void*)sVec3);
        glVertexAttribPointer(2,2,GL_FLOAT,GL_FALSE,stride,(void*)(2*sVec3));
        delete[] vtxData;
    }
};


class Torus : public ParamSurface {
public:
    vec4 center, ka, ks;
    float r, R;
    float shine, darab;
    mat4 M, Minv;
    Torus(vec4 c, float r, float R, float darab) :
    center(c),
    darab(darab),
    ka(vec4(0.2, 0.2, 0.2, 1)),
    ks(vec4(0.2, 0.2, 0.2, 1)),
    shine(0.01),
    r(r),
    R(R) {
    }
    
    void Create(){
        ParamSurface::Create(64, 64);
    }
    
    VertexData GenVertexData(float u, float v) { //az u-t es v-t felcsereltem, de keson vettem eszre.
        VertexData vd;
        vec4 tan1 = vec4(-cosf(u*2*M_PI)*sinf(v*2*M_PI),
                         -sinf(v*2*M_PI)*sinf(u*2*M_PI),
                         cosf(v*2*M_PI)).normalize();
        vec4 tan2 = vec4(-sinf(u*2*M_PI),
                         cosf(u*2*M_PI),
                         0).normalize();
        vd.normal = cross(tan1, tan2);
        
        vd.position = vec4((R+r*cosf(v*2*M_PI))*cosf(u*2*M_PI),
                           (R+r*cosf(v*2*M_PI))*sinf(u*2*M_PI),
                           r*sinf(v*2*M_PI));
        vd.u = u;
        vd.v = v;
        return vd;
    }
    
    void Draw() {
        M = Scale(1, 1, 1) *
        Rotate(-M_PI/2.0, 1, 0, 0) *
        Translate(0, 0, 0);
        Minv = Translate(0, 0, 0) *
        Rotate(M_PI/2.0, 1,0,0) *
        Scale(1, 1, 1);
        mat4 MVP = M * camera.V() * camera.P();
        
        M.SetUniform(Phong.shaderProgram, "M");
        MVP.SetUniform(Phong.shaderProgram, "MVP");
        Minv.SetUniform(Phong.shaderProgram, "Minv");
        ka.SetUniform(Phong.shaderProgram, "ka");
        ks.SetUniform(Phong.shaderProgram, "ks");
        int location = glGetUniformLocation(Phong.shaderProgram, "shine");
        if (location >= 0) glUniform1fv(location, 1, &shine);
        else printf("shine cannot be set\n");
        location = glGetUniformLocation(Phong.shaderProgram, "darab");
        if (location >= 0) glUniform1fv(location, 1, &darab);
        else printf("darab cannot be set\n");
        
        Geometry::Draw();
    }
};

class Sphere : public ParamSurface {
    vec4 center, ka, ks;
    float radius, shine, darab, rotate = 0, insideRotate = 0, outsideRotate = 0, moveR, mover;
    mat4 M;
    mat4 rotationMatrix, inverseRotationMatrix;
public:
    vec4 lastPos;
    Sphere(vec4 c, float r, float darab) :
    radius(r),
    darab(darab),
    ka(vec4(0.2,0.2,0.2,1)),
    ks(vec4(0.2,0.2,0.2,1)),
    shine(0.5),
    center(vec4())
    { }
    
    void Create(){
        ParamSurface::Create(64, 32);
        rotationMatrix = Scale(1, 1, 1);
        inverseRotationMatrix = Scale(1, 1, 1);
    }
    
    VertexData GenVertexData(float u, float v) {
        VertexData vd;
        vd.normal = vec4(cos(u*2*M_PI) * sin(v*M_PI),
                         sin(u*2*M_PI) * sin(v*M_PI),
                         cos(v*M_PI));
        vd.position = vd.normal * radius + center;
        vd.u = u; vd.v = v;
        return vd;
    }
    
    void Draw() {
        
        M = Scale(1, 1, 1) *
        rotationMatrix *
        Translate(lastPos.x, lastPos.y, lastPos.z);
        mat4 Minv = Translate(-lastPos.x, -lastPos.y, -lastPos.z) *
        inverseRotationMatrix *
        Scale(1, 1, 1);
        mat4 MVP = M * camera.V() * camera.P();
        
        M.SetUniform(Phong.shaderProgram, "M");
        Minv.SetUniform(Phong.shaderProgram, "Minv");
        MVP.SetUniform(Phong.shaderProgram, "MVP");
        
        ka.SetUniform(Phong.shaderProgram, "ka");
        ks.SetUniform(Phong.shaderProgram, "ks");
        int location = glGetUniformLocation(Phong.shaderProgram, "shine");
        if (location >= 0) glUniform1fv(location, 1, &shine);
        else printf("shine cannot be set\n");
        location = glGetUniformLocation(Phong.shaderProgram, "darab");
        if (location >= 0) glUniform1fv(location, 1, &darab);
        else printf("darab cannot be set\n");
        glBindVertexArray(vao);
        glDrawArrays(GL_TRIANGLES, 0, nVtx);
    }
    void Animate(float t, Torus torus) {

        float x = 2;
        t = t/3;
        t = t+0.555f;
        
        VertexData vd = torus.GenVertexData(t, sin(x*t));
        rotate = 0;
        vec4 realNormal = (vd.normal.normalize() * Rotate(-M_PI/2, 1, 0, 0)).normalize();
        vec4 currPos = vd.position * Rotate(-M_PI/2, 1, 0, 0);
        float R = torus.R;
        float r = torus.r;
        vec4 forward = vec4(-2*M_PI*sinf(2*M_PI*t)*(r*cosf(2*M_PI*sinf(t*x))+R)-2*M_PI*r*x*cosf(2*M_PI*t)*sinf(2*M_PI*sinf(t*x))*cosf(t*x),
                            2*M_PI*cosf(2*M_PI*t)*(r*cosf(2*M_PI*sinf(t*x))+R)-2*M_PI*r*x*sinf(2*M_PI*t)*sinf(2*M_PI*sinf(t*x))*cosf(t*x),
                            2*M_PI*r*x*cosf(t*x)*cosf(2*M_PI*sinf(t*x)),
                            1).normalize() *  Rotate(-M_PI/2, 1, 0, 0);
        forward = forward.normalize();
        currPos = currPos + (realNormal * radius);
        vec4 rotationAxis = cross(forward, realNormal).normalize();
        float distance = (lastPos - currPos).length();
        float kerulet = 2 * radius * M_PI;
        float rad = (distance/kerulet) * 2 * M_PI;
        
        rotationMatrix = rotationMatrix * Rotate(rad, rotationAxis.x, rotationAxis.y, rotationAxis.z);
        inverseRotationMatrix = inverseRotationMatrix * Rotate(-rad, rotationAxis.x, rotationAxis.y, rotationAxis.z);
        
        lastPos = currPos;
        
        
    }
};
class Light {
public:
    vec4 wLiPos;
    vec4 Le;
    vec4 La;
    vec4 dirVector;
    float lastT = 0;
    
    Light(vec4 pos, vec4 color, vec4 startDir) :wLiPos(pos), Le(color), dirVector(startDir) {
        La = vec4(0.1,0.1,0.1,1);
    }
    
    virtual void Create() = 0;
    
    void Animate(float tt, Torus torus) {
        vec4 ray = dirVector.normalize();
        vec4 d = ray;
        vec4 p = wLiPos;
        
        //ray egyenletet behelyettesitem a torusz egyenletebe
        float R = torus.R;
        float r = torus.r;
        float pz = p.y;
        float dz = ray.y;
        float alfa = dot(d, d);
        float beta = 2*dot(p, d);
        float gamma = dot(p, p) - (r*r) - (R*R);
        float a = alfa * alfa;
        float b = 2 * alfa * beta;
        float c = (beta * beta) + (2*alfa*beta) + (4*R*R*dz*dz);
        float da = (2*beta*gamma) + (8*R*R*pz*dz);
        float e = (gamma*gamma) + (4*R*R*pz*pz) - (4*R*R*r*r);
        
        float t = 0.00001f;
        //itt egy negyedfoku egyenletet kapok, amit kozelitessel szamolok. akkor lesz pontosan 0 az erteke a resultnak, ha a hitpointban vagyok
        float result = a * t * t * t * t + b * t * t * t + c * t * t + da * t + e;
        int defsign = result < 0 ? -1 : 1;
        while(t < 1) //ha t nagyobb mint 1, akkor nem vagyok kivancsi a metszespontra
        {
            result = (a * t * t * t * t + b * t * t * t + c * t * t + da * t + e);
            t += 0.00001f;
            int ressign = result < 0 ? -1 : 1;
            if(ressign != defsign)
                break;
        }
        
        //printf("%lf\n", t);
        if(t < 0.1)
        {
            vec4 hitPoint = p + d * t;
            float x = hitPoint.x;
            float y = hitPoint.y;
            float z = hitPoint.z;
            vec4 normal = vec4(4*x*(x*x + y*y + z*z - r*r - R*R),
                               4*y*(x*x + y*y + z*z - r*r - R*R) + (8 * R * R * y),
                               4*z*(x*x + y*y + z*z - r*r - R*R)).normalize() * -1;
            dirVector = reflect(dirVector, normal);
            dirVector = dirVector;
        }
        
        
        vec4 g = vec4(0,-1.0,0) * (tt-lastT);
        dirVector = dirVector + g;
        
        wLiPos = wLiPos + ((dirVector) * (tt-lastT));
        
        lastT = tt;
        
        this->Create();
        
        
    }
};

class YellowLight : public Light
{
public:
    YellowLight(vec4 pos, vec4 color, vec4 startDir) : Light(pos, color, startDir) { }
    void Create()
    {
        wLiPos.SetUniform(Phong.shaderProgram, "YwLiPos");
        Le.SetUniform(Phong.shaderProgram, "YLe");
        La.SetUniform(Phong.shaderProgram, "YLa");
    }
};

class CyanLight : public Light
{
public:
    CyanLight(vec4 pos, vec4 color, vec4 startDir) : Light(pos, color, startDir) { }
    void Create()
    {
        wLiPos.SetUniform(Phong.shaderProgram, "CwLiPos");
        Le.SetUniform(Phong.shaderProgram, "CLe");
        La.SetUniform(Phong.shaderProgram, "CLa");
    }
};

Torus torus = Torus(vec4(), 2, 5, 40);
Sphere sphere = Sphere(vec4(1.5,0,15), 0.7f, 4);
YellowLight lightYellow = YellowLight(vec4(5.f, -1.0f, -3.f), vec4(1,1,0.4,1), vec4(-2, +2, -1.2, 1).normalize() * 2.5f);
CyanLight lightCyan = CyanLight(vec4(5.8f, -1.0f, -2.f), vec4(0.4,1,1,1), vec4(-4, +1.0, -1.1, 1).normalize() * 2.0f);

void onInitialization() {
    glViewport(0, 0, windowWidth, windowHeight);
    const char* arr[] = { "vtxPos", "vtxNormal", "uv" };
    Phong.Create(vertexSource, arr, 3, fragmentSource, "fragmentColor");
    Phong.Bind();
    
    texture.Create();
    torus.Create();
    lightCyan.Create();
    lightYellow.Create();
    sphere.Create();
    camera.Create();
}

void onExit() {
    glDeleteProgram(Phong.shaderProgram);
}

void onDisplay() {
    glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    glDepthFunc(GL_LESS);
    torus.Draw();
    sphere.Draw();
    glutSwapBuffers();
}

void onKeyboard(unsigned char key, int pX, int pY) {
    
}

void onKeyboardUp(unsigned char key, int pX, int pY) {
    
}

void onMouse(int button, int state, int pX, int pY) {
}

void onMouseMotion(int pX, int pY) {
}

float pT = 0;
void onIdle() {
    long time = glutGet(GLUT_ELAPSED_TIME);
    float rT = time / 1000.0f;
    
    float interval = 0.01f;
    for (float sec = pT; sec < rT; sec+=interval) {
        camera.Animate(sec);
        lightYellow.Animate(sec, torus);
        lightCyan.Animate(sec, torus);
        sphere.Animate(sec, torus);
    }
    pT = rT;
    glutPostRedisplay();
}

// Idaig modosithatod...
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

int main(int argc, char * argv[]) {
    glutInit(&argc, argv);
#if !defined(__APPLE__)
    glutInitContextVersion(majorVersion, minorVersion);
#endif
    glutInitWindowSize(windowWidth, windowHeight);				// Application window is initially of resolution 600x600
    glutInitWindowPosition(100, 100);							// Relative location of the application window
#if defined(__APPLE__)
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH | GLUT_3_2_CORE_PROFILE);  // 8 bit R,G,B,A + double buffer + depth buffer
#else
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
#endif
    glutCreateWindow(argv[0]);
    
#if !defined(__APPLE__)
    glewExperimental = true;	// magic
    glewInit();
#endif
    
    printf("GL Vendor    : %s\n", glGetString(GL_VENDOR));
    printf("GL Renderer  : %s\n", glGetString(GL_RENDERER));
    printf("GL Version (string)  : %s\n", glGetString(GL_VERSION));
    glGetIntegerv(GL_MAJOR_VERSION, &majorVersion);
    glGetIntegerv(GL_MINOR_VERSION, &minorVersion);
    printf("GL Version (integer) : %d.%d\n", majorVersion, minorVersion);
    printf("GLSL Version : %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));
    
    onInitialization();
    
    glutDisplayFunc(onDisplay);                // Register event handlers
    glutMouseFunc(onMouse);
    glutIdleFunc(onIdle);
    glutKeyboardFunc(onKeyboard);
    glutKeyboardUpFunc(onKeyboardUp);
    glutMotionFunc(onMouseMotion);
    
    glutMainLoop();
    onExit();
    return 1;
}