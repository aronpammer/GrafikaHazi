//
//  main.cpp
//  ElsoHazi
//
//  Created by Pammer Áron on 12/03/16.
//  Copyright (c) 2016 Aron Pammer. All rights reserved.
//

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
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
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
#include <GL/glew.h>		// must be downloaded
#include <GL/freeglut.h>	// must be downloaded unless you have an Apple
#endif

const unsigned int windowWidth = 600, windowHeight = 600;

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Innentol modosithatod...

#define maxControlPoints        10
#define splineSmoothing         20
#define firstLastBetween        0.5f
#define edgeOfStar              8
#define G                       2.0f
#define nu                      0.5f
#define DOPPLER                 true
#define C                       30
#define m1                      1
#define m2                      10
#define GRAVITATIONAL_PRECISION 0.0001f


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
#version 140
precision highp float;

uniform mat4 MVP;

in vec2 vertexPosition;
in vec3 vertexColor;
out vec3 color;

void main() {
    color = vertexColor;
    gl_Position = vec4(vertexPosition.x, vertexPosition.y, 0, 1) * MVP;
}
)";

const char *fragmentSource = R"(
#version 140
precision highp float;

in vec3 color;
out vec4 fragmentColor;

void main() {
    fragmentColor = vec4(color, 1);
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
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                result.m[i][j] = 0;
                for (int k = 0; k < 4; k++) result.m[i][j] += m[i][k] * right.m[k][j];
            }
        }
        return result;
    }
    operator float*() { return &m[0][0]; }
};

struct vec4 {
    float v[4];
    
    
    vec4(float x = 0, float y = 0, float z = 0, float w = 1) {
        v[0] = x; v[1] = y; v[2] = z; v[3] = w;
    }
    
    vec4(const vec4& v)
    {
        this->v[0] = v.v[0];
        this->v[1] = v.v[1];
        this->v[2] = v.v[2];
        this->v[3] = v.v[3];
    }
    
    vec4& operator=(const vec4& v)
    {
        this->v[0] = v.v[0];
        this->v[1] = v.v[1];
        this->v[2] = v.v[2];
        this->v[3] = v.v[3];
        return *this;
    }
    
    vec4 operator*(const mat4& mat) {
        vec4 result;
        for (int j = 0; j < 4; j++) {
            result.v[j] = 0;
            for (int i = 0; i < 4; i++) result.v[j] += v[i] * mat.m[i][j];
        }
        return result;
    }
    vec4 operator*(const float& n) {
        vec4 result;
        result.v[0] = this->v[0] * n;
        result.v[1] = this->v[1] * n;
        result.v[2] = this->v[2] * n;
        result.v[3] = this->v[3] * n;
        return result;
    }
    
    vec4 operator/(const float& n) {
        if(n == 0)
            return vec4();
        vec4 result;
        result.v[0] = this->v[0] / n;
        result.v[1] = this->v[1] / n;
        result.v[2] = this->v[2] / n;
        result.v[3] = this->v[3] / n;
        return result;
    }
    
    vec4 operator+(const vec4& n) {
        vec4 result;
        result.v[0] = n.v[0] + this->v[0];
        result.v[1] = n.v[1] + this->v[1];
        result.v[2] = n.v[2] + this->v[2];
        result.v[3] = n.v[3] + this->v[3];
        return result;
    }
    
    vec4 operator-(const vec4& n) {
        vec4 result;
        result.v[0] = this->v[0] - n.v[0];
        result.v[1] = this->v[1] - n.v[1];
        result.v[2] = this->v[2] - n.v[2];
        result.v[3] = this->v[3] - n.v[3];
        return result;
    }
    
};

struct Camera {
    float wCx, wCy;
    float wWx, wWy;
    bool follow;
public:
    Camera() {
        follow = false;
        Animate(0, 0, 0);
    }
    
    mat4 V() {
        return mat4(1,    0, 0, 0,
                    0,    1, 0, 0,
                    0,    0, 1, 0,
                    -wCx, -wCy, 0, 1);
    }
    
    mat4 P() {
        return mat4(2/wWx,    0, 0, 0,
                    0,    2/wWy, 0, 0,
                    0,        0, 1, 0,
                    0,        0, 0, 1);
    }
    
    mat4 Vinv() {
        return mat4(1,     0, 0, 0,
                    0,     1, 0, 0,
                    0,     0, 1, 0,
                    wCx, wCy, 0, 1);
    }
    
    mat4 Pinv() {
        return mat4(wWx/2, 0,    0, 0,
                    0, wWy/2, 0, 0,
                    0,  0,    1, 0,
                    0,  0,    0, 1);
    }
    void getPosition(float* x, float* y)
    {
        *x = wCx;
        *y = wCy;
    }
    void toggleFollow()
    {
        follow = !follow;
    }
    void Animate(float t, float followX, float followY) {
    
        if(follow)
        {
            wCx = followX;
            wCy = followY;
        }
        else
        {
            wCx = 0;
            wCy = 0;
        }
        wWx = 20;
        wWy = 20;
    }
};

Camera camera;

unsigned int shaderProgram;

class CatmullRom {
    
    float initT;
    
    vec4 Hermite( vec4 p0, vec4 v0, float t0,
                 vec4 p1, vec4 v1, float t1,
                 float t )
    {
        vec4 a0 = p0;
        vec4 a1 = v0;
        vec4 a2 = (((p1 - p0) * 3) / ((t1 - t0) * (t1 - t0))) - ((v1 + (v0 * 2))/(t1-t0));
        vec4 a3 = (((p0 - p1) * 2) / ((t1 - t0) * (t1 - t0) * (t1 - t0))) + ((v1 + v0) / ((t1 - t0) * (t1 - t0)));
        vec4 ft = (a3 * (t-t0) * (t-t0) * (t-t0)) + (a2 * (t-t0) * (t-t0)) + (a1 * (t-t0)) + a0;
        
        return ft;
    }
    
public:
    vec4* cps;
    float* ts;
    vec4* vs;
    int cpc;
    
    CatmullRom()
    {
        cps = new vec4[maxControlPoints + 1];
        ts = new float[maxControlPoints + 1];
        vs = new vec4[maxControlPoints + 1];
        cpc = 0;
    }
    ~CatmullRom()
    {
        delete[] cps;
        delete[] ts;
        delete[] vs;
    }
    void AddControlPoint(vec4 cp, float t)
    {
        if(cpc == 0)
            initT = t;
        cps[cpc] = cp;
        ts[cpc] = t - initT;
        cpc++;
        calcVs();
    }
    void calcVs()
    {
        if(cpc == 1)
            vs[0] = 0;
        else
        {
            vs[0] = (((cps[1] - cps[0]) / (ts[1] - ts[0])) + ((cps[0] - cps[cpc-1]) / (firstLastBetween))) * 0.9f;
        }
        
        for (int i = 1; i < cpc - 1; i++) {
            vs[i] = (((cps[i+1] - cps[i]) / (ts[i+1] - ts[i])) + ((cps[i] - cps[i-1]) / (ts[i] - ts[i-1])));
            vs[i] = vs[i] * 0.9f;
        }
        vs[cpc-1] = (((cps[0] - cps[cpc-1]) / (firstLastBetween)) + ((cps[cpc-1] - cps[cpc-2]) / (ts[cpc-1] - ts[cpc-2]))) * 0.9f;
    }
    
    float getMaxT()
    {
        if(cpc == 0)
            return 0;
        return ts[cpc-1] + firstLastBetween;
    }
   
    vec4 r(float st) {
        if(cpc <= 1)
            return vec4();
        
        float t = st - (((int)(st / getMaxT())) * getMaxT());

        
        for(int i = 0; i < cpc - 1; i++) {
            if (ts[i] <= t && t <= ts[i+1]) return Hermite(cps[i], vs[i], ts[i], cps[i + 1], vs[i + 1], ts[i + 1], t);
        }
        
        return Hermite(cps[cpc-1], vs[cpc-1], ts[cpc-1], cps[0], vs[0], ts[cpc-1] + firstLastBetween, t);
        
    }
};


class LineStrip {
    GLuint vao, vbo;
    float*  splineVertexData;
    int    nVertices;
    CatmullRom cr;
    int totalDraw;
public:
    LineStrip() {
        nVertices = 0;
        splineVertexData = new float[5*((splineSmoothing*maxControlPoints)+1)];
    }
    ~LineStrip()
    {
        delete[] splineVertexData;
    }
    void Create() {
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);
        
        glGenBuffers(1, &vbo);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glEnableVertexAttribArray(0);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), reinterpret_cast<void*>(0));
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), reinterpret_cast<void*>(2 * sizeof(float)));
    }
    vec4 getPoint(float t)
    {
        return cr.r(t);
    }
    float getMaxT()
    {
        return cr.getMaxT();
    }
    void AddPoint(float cX, float cY) {
        
        if (nVertices >= maxControlPoints) return;
        
        vec4 wVertex = vec4(cX, cY, 0, 1) * camera.Pinv() * camera.Vinv();
        
        long time = glutGet(GLUT_ELAPSED_TIME);
        float sec = time / 1000.0f;
        cr.AddControlPoint(wVertex, sec);
        nVertices++;
        int numberOfVertexData = 0;
        
        for (int i = 0; i < cr.cpc-1; i++) {
            for (float t = cr.ts[i]; t <= cr.ts[i+1]; t+= (cr.ts[i+1] - cr.ts[i])/splineSmoothing)
            {
                splineVertexData[5 * numberOfVertexData]     = cr.r(t).v[0];
                splineVertexData[5 * numberOfVertexData + 1] = cr.r(t).v[1];
                splineVertexData[5 * numberOfVertexData + 2] = 0;
                splineVertexData[5 * numberOfVertexData + 3] = 1;
                splineVertexData[5 * numberOfVertexData + 4] = 0;
                numberOfVertexData++;
            }
        }
        for (float t = cr.ts[cr.cpc-1]; t <= cr.getMaxT(); t+= firstLastBetween/splineSmoothing) {
            splineVertexData[5 * numberOfVertexData]     = cr.r(t).v[0];
            splineVertexData[5 * numberOfVertexData + 1] = cr.r(t).v[1];
            splineVertexData[5 * numberOfVertexData + 2] = 0;
            splineVertexData[5 * numberOfVertexData + 3] = 1;
            splineVertexData[5 * numberOfVertexData + 4] = 0;
            numberOfVertexData++;
        }
        splineVertexData[5 * numberOfVertexData]     = cr.r(cr.getMaxT()).v[0];
        splineVertexData[5 * numberOfVertexData + 1] = cr.r(cr.getMaxT()).v[1];
        splineVertexData[5 * numberOfVertexData + 2] = 0;
        splineVertexData[5 * numberOfVertexData + 3] = 1;
        splineVertexData[5 * numberOfVertexData + 4] = 0;
        
        totalDraw = (numberOfVertexData+1) * 5;
        
        
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, totalDraw * sizeof(float), splineVertexData, GL_DYNAMIC_DRAW);
    }
    
    void Draw() {
        if (nVertices > 0) {
            mat4 VPTransform = camera.V() * camera.P();
            
            int location = glGetUniformLocation(shaderProgram, "MVP");
            if (location >= 0) glUniformMatrix4fv(location, 1, GL_TRUE, VPTransform);
            else printf("uniform MVP cannot be set\n");
            
            glBindVertexArray(vao);
        
            glDrawArrays(GL_LINE_STRIP, 0, totalDraw/5);
            
            
        }
        
    }
};

LineStrip lineStrip;


class Star
{
    float sx, sy;
    float wTx, wTy;
    float rAngle;
    unsigned int vao;
    unsigned int vbo[2];
    int numberOfVertices;
    vec4 V0;
    vec4 color;
    vec4 origcolor;
    float radius;
    float innerT;
    float lastwTx, lastwTy;
public:
    Star(vec4 color, float radius)
    {
        this->color = color;
        this->origcolor = color;
        this->radius = radius;
        this->innerT = 0;
        this->V0 = vec4(0,15,0);
    }
    void setPosition(float x, float y)
    {
        wTx = x;
        wTy = y;
    }
    void getPosition(float* x, float* y)
    {
        *x = wTx;
        *y = wTy;
    }
    float* getStarCoords(int edges, int* size)
    {
        *size = edges * 12;
        float* coords = new float[*size];
        
        float basedegree = 360.0f/edges;
        int coordc = 0;
        float lastx = radius;
        float lasty = 0;
        for (int i = 1; i <= edges; i++) {
            float degree = ((i*basedegree)/180) * M_PI;
            float x = cosf(degree) * radius;
            float y = sinf(degree) * radius;
            
            coords[coordc++] = 0;
            coords[coordc++] = 0;
            coords[coordc++] = lastx;
            coords[coordc++] = lasty;
            coords[coordc++] = x;
            coords[coordc++] = y;
            
            
            float kdegree = (((i*basedegree)-(basedegree/2))/180) * M_PI;
            
            float kix = cosf(kdegree) * (radius*2);
            float kiy = sinf(kdegree) * (radius*2);
            
            coords[coordc++] = lastx;
            coords[coordc++] = lasty;
            coords[coordc++] = x;
            coords[coordc++] = y;
            coords[coordc++] = kix;
            coords[coordc++] = kiy;
            
            
            lastx = x;
            lasty = y;
        }
        
        
        
        return coords;
    }
    float* getStarColors(int vertices)
    {
        float* colors = new float[vertices*3];
        int colorc = 0;
        for (int i = 0; i < vertices; i++) {
            colors[colorc++] = this->color.v[0];
            colors[colorc++] = this->color.v[1];
            colors[colorc++] = this->color.v[2];
        }
        return colors;
    }
    
    void changeColor(vec4 color)
    {
        this->color = color;
        
    }
    
    void Create()
    {
        int size;
        float* vertexCoords = getStarCoords(edgeOfStar, &size);
        numberOfVertices = size / 2;
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);
        glGenBuffers(2, &vbo[0]);
        glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
        glBufferData(GL_ARRAY_BUFFER,
                     size * sizeof(GLfloat),
                     vertexCoords,
                     GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0,
                              2, GL_FLOAT,
                              GL_FALSE,
                              0, NULL);
        glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);
        float* vertexColors = getStarColors(numberOfVertices);
        glBufferData(GL_ARRAY_BUFFER, numberOfVertices * 3 * sizeof(GLfloat), vertexColors, GL_STATIC_DRAW);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, NULL);
        
        delete[] vertexCoords;
        delete[] vertexColors;
    }
    void Animate(float t, Star* follow, float pastT)
    {
        if(follow == nullptr)
        {
            innerT += pastT;
            if(innerT > lineStrip.getMaxT())
            {
                innerT = innerT - lineStrip.getMaxT();
            }
            
            vec4 v4 = lineStrip.getPoint(innerT);
            wTx = v4.v[0];
            wTy = v4.v[1];
        }
        else
        {
            float owTx;
            float owTy;
            follow->getPosition(&owTx, &owTy);
            
            vec4 S;
            float val = GRAVITATIONAL_PRECISION;
            for (float i = 0; i <= pastT; i += val) {
                vec4 wD(owTx - wTx, owTy - wTy, 0, 0);
                float distance = sqrtf((wD.v[0] * wD.v[0]) + (wD.v[1] * wD.v[1]));
                vec4 normalized = wD / distance;
                float norm_dist = sqrtf((normalized.v[0] * normalized.v[0]) + (normalized.v[1] * normalized.v[1]));
                vec4 F;
                if(norm_dist != 0)
                   F = (normalized * (G * ((m1 * m2) / (norm_dist*norm_dist*norm_dist)))) - (V0 * nu);
                
                V0 = V0 + (F * val);
                S = (V0 * val);
                
                wTx += S.v[0];
                wTy += S.v[1];
            }
        }
        if(DOPPLER)
        {
            float distFromOrigo = sqrtf((wTx * wTx) + (wTy * wTy));
            float secDistFromOrigo = sqrtf((lastwTx * lastwTx) + (lastwTy * lastwTy));
            float V = (secDistFromOrigo - distFromOrigo) / pastT;
            float rate = V / C;
            float newRed = origcolor.v[0] + rate;
            if(newRed > 1)
                newRed = 1;
            float newBlue = origcolor.v[2] - rate;
            if(newBlue < 0)
                newBlue = 0;
            vec4 newColor(newRed, origcolor.v[1], newBlue);
            this->changeColor(newColor);
        }
        lastwTx = wTx;
        lastwTy = wTy;
        
        sx = 2 + (0.5 * sinf(t * 5));
        sy = 2 + (0.5 * sinf(t * 5));
        
        rAngle = t;
    }
    void Draw()
    {
        if(DOPPLER == true)
        {
            glBindVertexArray(vao);
            glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);
            float* vertexColors = getStarColors(numberOfVertices);
            glBufferData(GL_ARRAY_BUFFER, numberOfVertices * 3 * sizeof(GLfloat), vertexColors, GL_STATIC_DRAW);
            glEnableVertexAttribArray(1);
            glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, NULL);
            delete[] vertexColors;
        }
        
        
        mat4 M(sx,   0,  0, 0,
               0,  sy,  0, 0,
               0,   0,  0, 0,
               wTx, wTy,  0, 1);
        mat4 Mrot(cosf(rAngle), -sinf(rAngle), 0, 0,
                  sinf(rAngle), cosf(rAngle),  0, 0,
                  0,           0,            1, 0,
                  0,           0,            0, 1);
        
        mat4 MVPTransform = Mrot * M * camera.V() * camera.P();
        int location = glGetUniformLocation(shaderProgram, "MVP");
        if (location >= 0) glUniformMatrix4fv(location, 1, GL_TRUE, MVPTransform);
        else printf("uniform MVP cannot be set\n");
        
        glBindVertexArray(vao);
        
        glDrawArrays(GL_TRIANGLES, 0, numberOfVertices);
        
        
    }
};

Star star(vec4(0.6f, 0.2f,0.1f), 0.5);
Star star2(vec4(0.3f,0.1f,0.1f), 0.25f);
Star star3(vec4(0.4f,0,0.2f), 0.25f);

void onInitialization() {
    glViewport(0, 0, windowWidth, windowHeight);
    star.Create();
    star2.Create();
    star2.setPosition(-3, 0);
    star3.Create();
    star3.setPosition(3, 0);
    lineStrip.Create();
    unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
    if (!vertexShader) {
        printf("Error in vertex shader creation\n");
        exit(1);
    }
    glShaderSource(vertexShader, 1, &vertexSource, NULL);
    glCompileShader(vertexShader);
    checkShader(vertexShader, "Vertex shader error");
    
    unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    if (!fragmentShader) {
        printf("Error in fragment shader creation\n");
        exit(1);
    }
    glShaderSource(fragmentShader, 1, &fragmentSource, NULL);
    glCompileShader(fragmentShader);
    checkShader(fragmentShader, "Fragment shader error");
    
    shaderProgram = glCreateProgram();
    if (!shaderProgram) {
        printf("Error in shader program creation\n");
        exit(1);
    }
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    
    glBindAttribLocation(shaderProgram, 0, "vertexPosition");
    glBindAttribLocation(shaderProgram, 1, "vertexColor");
    
    glBindFragDataLocation(shaderProgram, 0, "fragmentColor");
    
    glLinkProgram(shaderProgram);
    checkLinking(shaderProgram);
    glUseProgram(shaderProgram);
}

void onExit() {
    glDeleteProgram(shaderProgram);
    printf("exit");
}

void onDisplay() {
    glClearColor(0.2, 0.2, 0.2, 0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    
    star.Draw();
    star2.Draw();
    star3.Draw();
    lineStrip.Draw();
    
    glutSwapBuffers();
}

void onKeyboard(unsigned char key, int pX, int pY) {
    if (key == 'd') glutPostRedisplay();
    if(key == ' ')
    {
        camera.toggleFollow();
    }
}

void onKeyboardUp(unsigned char key, int pX, int pY) {
    
}

void onMouse(int button, int state, int pX, int pY) {
    if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {
        float cX = 2.0f * pX / windowWidth - 1;
        float cY = 1.0f - 2.0f * pY / windowHeight;
        lineStrip.AddPoint(cX, cY);
        glutPostRedisplay();
    }
}

void onMouseMotion(int pX, int pY) {
}
float lasts = 0;
void onIdle() {
    long time = glutGet(GLUT_ELAPSED_TIME);
    float sec = time / 1000.0f;
    star.Animate(sec, nullptr, sec - lasts);
    star2.Animate(sec, &star, sec - lasts);
    star3.Animate(sec, &star, sec - lasts);
    float x,y;
    star.getPosition(&x, &y);
    camera.Animate(sec, x, y);
    glutPostRedisplay();
    
    lasts = sec;
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

