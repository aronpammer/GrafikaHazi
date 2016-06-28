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

#define maxControlPoints        10     // maximum control points
#define splineSmoothing         20      // smoothing of the spline
#define firstLastBetween        0.5f    // How many secs between the first and the last control point
#define edgeOfStar              8       // number of edges of the Star
#define G                       2.0f    // gravitational constant
#define nu                      0.5f    // slow down the small stars by a factor
#define DOPPLER                 true    // is doppler enabled
#define C                       30      // relative speed of light. factor to make doppler effect work.
#define m1                      1       // weight of small stars
#define m2                      10      // weight of big star
#define GRAVITATIONAL_PRECISION 0.0001f


// OpenGL major and minor versions
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

// check if shader could be compiled
void checkShader(unsigned int shader, char * message) {
    int OK;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &OK);
    if (!OK) {
        printf("%s!\n", message);
        getErrorInfo(shader);
    }
}

// check if shader could be linked
void checkLinking(unsigned int program) {
    int OK;
    glGetProgramiv(program, GL_LINK_STATUS, &OK);
    if (!OK) {
        printf("Failed to link shader program!\n");
        getErrorInfo(program);
    }
}

// vertex shader in GLSL
const char *vertexSource = R"(
#version 140
precision highp float;

uniform mat4 MVP;			// Model-View-Projection matrix in row-major format

in vec2 vertexPosition;		// variable input from Attrib Array selected by glBindAttribLocation
in vec3 vertexColor;	    // variable input from Attrib Array selected by glBindAttribLocation
out vec3 color;				// output attribute

void main() {
    color = vertexColor;														// copy color from input to output
    gl_Position = vec4(vertexPosition.x, vertexPosition.y, 0, 1) * MVP; 		// transform to clipping space
}
)";

// fragment shader in GLSL
const char *fragmentSource = R"(
#version 140
precision highp float;

in vec3 color;				// variable input: interpolated color of vertex shader
out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation

void main() {
    fragmentColor = vec4(color, 1); // extend RGB to RGBA
}
)";

// row-major matrix 4x4
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

// 3D point in homogeneous coordinates
struct vec4 {
    float v[4];
    
    
    vec4(float x = 0, float y = 0, float z = 0, float w = 1) {
        v[0] = x; v[1] = y; v[2] = z; v[3] = w;
    }
    //copy constructor
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
// 2D camera
struct Camera {
    float wCx, wCy;	// center in world coordinates
    float wWx, wWy;	// width and height in world coordinates
    bool follow;    // follow star or not
public:
    Camera() {
        follow = false;
        Animate(0, 0, 0);
    }
    
    mat4 V() { // view matrix: translates the center to the origin
        return mat4(1,    0, 0, 0,
                    0,    1, 0, 0,
                    0,    0, 1, 0,
                    -wCx, -wCy, 0, 1);
    }
    
    mat4 P() { // projection matrix: scales it to be a square of edge length 2
        return mat4(2/wWx,    0, 0, 0,
                    0,    2/wWy, 0, 0,
                    0,        0, 1, 0,
                    0,        0, 0, 1);
    }
    
    mat4 Vinv() { // inverse view matrix
        return mat4(1,     0, 0, 0,
                    0,     1, 0, 0,
                    0,     0, 1, 0,
                    wCx, wCy, 0, 1);
    }
    
    mat4 Pinv() { // inverse projection matrix
        return mat4(wWx/2, 0,    0, 0,
                    0, wWy/2, 0, 0,
                    0,  0,    1, 0,
                    0,  0,    0, 1);
    }
    // returns the camera's position
    void getPosition(float* x, float* y)
    {
        *x = wCx;
        *y = wCy;
    }
    // when space is pressed this function is called
    void toggleFollow()
    {
        follow = !follow;
    }
    // t: time
    // followX: X position of the object to follow
    // followY: Y position of the object to follow
    void Animate(float t, float followX, float followY) {
    
        if(follow)
        {
            wCx = followX;
            wCy = followY;
        }
        else
        {
            wCx = 0; // 10 * cosf(t);
            wCy = 0;
        }
        wWx = 20;
        wWy = 20;
    }
};

// 2D camera
Camera camera;

// handle of the shader program
unsigned int shaderProgram;

// CatmullRom class
class CatmullRom {
    
    float initT; // base T to start from, that is the first control point's time value. this is needed because the catmull rom has to start from 0 seconds. further explanation below
    
    // catmull rom's Hermite function as seen on the slides
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
    
    //init catmullrom
    CatmullRom()
    {
        cps = new vec4[maxControlPoints + 1];
        ts = new float[maxControlPoints + 1];
        vs = new vec4[maxControlPoints + 1];
        cpc = 0;
    }
    //cleaning up
    ~CatmullRom()
    {
        delete[] cps;
        delete[] ts;
        delete[] vs;
    }
    //adding a control point
    // cp: the position of the control point
    // t: time value of the related control point
    void AddControlPoint(vec4 cp, float t)
    {
        if(cpc == 0) //if there aren't any control points yet then the initT will store the time of the first control point
            initT = t;
        cps[cpc] = cp;
        ts[cpc] = t - initT;
        cpc++;
        calcVs();
    }
    //recalculate all velocity vectors. there may be a better solution to this, for example it is not required to recalculate ALL of the velocities just the last few of them
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
            vs[i] = vs[i] * 0.9f; // it is multiplied by 0.9 as requested on the subject's website
        }
        vs[cpc-1] = (((cps[0] - cps[cpc-1]) / (firstLastBetween)) + ((cps[cpc-1] - cps[cpc-2]) / (ts[cpc-1] - ts[cpc-2]))) * 0.9f;
    }
    
    float getMaxT()
    {
        if(cpc == 0)
            return 0;
        return ts[cpc-1] + firstLastBetween; //get the maximum time value of the spline. @firstLastBetween is added because it was specified that there has to be a time gap of 0.5 seconds between the LAST and the FIRST control point
    }
   
    vec4 r(float st) {
        if(cpc <= 1)
            return vec4();
        
        float t = st - (((int)(st / getMaxT())) * getMaxT()); // remainder of the time. for example if the spline is 5 seconds long and the current time is at 6 seconds, then the correct value to get is 1 second.

        for(int i = 0; i < cpc - 1; i++) {
            if (ts[i] <= t && t <= ts[i+1]) return Hermite(cps[i], vs[i], ts[i], cps[i + 1], vs[i + 1], ts[i + 1], t); //as seen on the slides
        }
        
        //here we are between the last and the first control point
        return Hermite(cps[cpc-1], vs[cpc-1], ts[cpc-1], cps[0], vs[0], ts[cpc-1] + firstLastBetween, t);
        
    }
};


class LineStrip {
    GLuint vao, vbo;            // vertex array object, vertex buffer object
    float*  splineVertexData;   // interleaved data of coordinates and colors
    int    nVertices;           // number of vertices
    CatmullRom cr;              // catmull rom class
    int totalDraw;              // total number of vertexdata to pass onto the gpu
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
        
        glGenBuffers(1, &vbo); // Generate 1 vertex buffer object
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        // Enable the vertex attribute arrays
        glEnableVertexAttribArray(0);  // attribute array 0
        glEnableVertexAttribArray(1);  // attribute array 1
        // Map attribute array 0 to the vertex data of the interleaved vbo
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), reinterpret_cast<void*>(0)); // attribute array, components/attribute, component type, normalize?, stride, offset
        // Map attribute array 1 to the color data of the interleaved vbo
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
        
        //If there are too many control points then return
        if (nVertices >= maxControlPoints) return;
        
        //Calculate click point related to the camera
        vec4 wVertex = vec4(cX, cY, 0, 1) * camera.Pinv() * camera.Vinv();
        
        long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program
        float sec = time / 1000.0f;				// convert msec to sec
        cr.AddControlPoint(wVertex, sec);       // add control point to the catmull rom
        nVertices++;                            // increase number of vertices
        int numberOfVertexData = 0;
        
        for (int i = 0; i < cr.cpc-1; i++) { //iterate through the control points
            for (float t = cr.ts[i]; t <= cr.ts[i+1]; t+= (cr.ts[i+1] - cr.ts[i])/splineSmoothing) //add @splineSmoothing points between the control points. This is so there are always the exact number of vertexes between each control point, hence it is easier to calculate how much memory the graphics card will need
            {
                splineVertexData[5 * numberOfVertexData]     = cr.r(t).v[0]; // get the exact point's X coordinate based on @t
                splineVertexData[5 * numberOfVertexData + 1] = cr.r(t).v[1]; // get the exact point's Y coordinate based on @t
                splineVertexData[5 * numberOfVertexData + 2] = 0;            // red
                splineVertexData[5 * numberOfVertexData + 3] = 1;            // green
                splineVertexData[5 * numberOfVertexData + 4] = 0;            // blue
                numberOfVertexData++;
            }
        }
        // Get the coordinates between the last and the first control point.
        for (float t = cr.ts[cr.cpc-1]; t <= cr.getMaxT(); t+= firstLastBetween/splineSmoothing) {
            splineVertexData[5 * numberOfVertexData]     = cr.r(t).v[0];
            splineVertexData[5 * numberOfVertexData + 1] = cr.r(t).v[1];
            splineVertexData[5 * numberOfVertexData + 2] = 0; // red
            splineVertexData[5 * numberOfVertexData + 3] = 1; // green
            splineVertexData[5 * numberOfVertexData + 4] = 0; // blue
            numberOfVertexData++;
        }
        // Get the coordinates for the last control point.
        splineVertexData[5 * numberOfVertexData]     = cr.r(cr.getMaxT()).v[0];
        splineVertexData[5 * numberOfVertexData + 1] = cr.r(cr.getMaxT()).v[1];
        splineVertexData[5 * numberOfVertexData + 2] = 0; // red
        splineVertexData[5 * numberOfVertexData + 3] = 1; // green
        splineVertexData[5 * numberOfVertexData + 4] = 0; // blue
        
        // totalDraw equals the number of "floats" in the splineVertexData array
        totalDraw = (numberOfVertexData+1) * 5;
        
        
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        // copy data to the GPU
        glBufferData(GL_ARRAY_BUFFER, totalDraw * sizeof(float), splineVertexData, GL_DYNAMIC_DRAW);
    }
    
    void Draw() {
        //Draw only if there are more than 1 control points
        if (nVertices > 0) {
            mat4 VPTransform = camera.V() * camera.P();
            
            int location = glGetUniformLocation(shaderProgram, "MVP");
            if (location >= 0) glUniformMatrix4fv(location, 1, GL_TRUE, VPTransform);
            else printf("uniform MVP cannot be set\n");
            
            glBindVertexArray(vao);
            
            // totalDraw / 5 : number of Vertexdata, including their X, Y position and color
            glDrawArrays(GL_LINE_STRIP, 0, totalDraw/5);
            
            
        }
        
    }
};

// The LineStrip object
LineStrip lineStrip;


class Star
{
    float sx, sy;		// scaling
    float wTx, wTy;		// translation
    float rAngle;       // rotation
    unsigned int vao;
    unsigned int vbo[2];		// vertex buffer objects
    int numberOfVertices;
    vec4 V0;            // speed of the star
    vec4 color;         // color of the star
    vec4 origcolor;         // color of the star
    float radius;       // radius of the star
    float innerT;       // current time related to the star
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
    // sets the position of the star
    void setPosition(float x, float y)
    {
        wTx = x;
        wTy = y;
    }
    // returns the position of the star
    void getPosition(float* x, float* y)
    {
        *x = wTx;
        *y = wTy;
    }
    // generates a star based on the number of edges and returns its' coordinates and the size required to store it
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
            float x = cos(degree) * radius;
            float y = sin(degree) * radius;
            
            //base part of it
            coords[coordc++] = 0;
            coords[coordc++] = 0;
            coords[coordc++] = lastx;
            coords[coordc++] = lasty;
            coords[coordc++] = x;
            coords[coordc++] = y;
            
            
            float kdegree = (((i*basedegree)-(basedegree/2))/180) * M_PI;
            
            float kix = cos(kdegree) * (radius*2);
            float kiy = sin(kdegree) * (radius*2);
            
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
    // generates a color vector
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
        
        // request a star.
        float* vertexCoords = getStarCoords(edgeOfStar, &size);
        
        numberOfVertices = size / 2;
        
        glGenVertexArrays(1, &vao);	// create 1 vertex array object
        glBindVertexArray(vao);		// make it active
        
        glGenBuffers(2, &vbo[0]);	// Generate 2 vertex buffer objects
        
        // vertex coordinates: vbo[0] -> Attrib Array 0 -> vertexPosition of the vertex shader
        glBindBuffer(GL_ARRAY_BUFFER, vbo[0]); // make it active, it is an array
        
        glBufferData(GL_ARRAY_BUFFER,      // copy to the GPU
                     size * sizeof(GLfloat), // number of the vbo in bytes
                     vertexCoords,		   // address of the data array on the CPU
                     GL_STATIC_DRAW);	   // copy to that part of the memory which is not modified
        // Map Attribute Array 0 to the current bound vertex buffer (vbo[0])
        glEnableVertexAttribArray(0);
        // Data organization of Attribute Array 0
        glVertexAttribPointer(0,			// Attribute Array 0
                              2, GL_FLOAT,  // components/attribute, component type
                              GL_FALSE,		// not in fixed point format, do not normalized
                              0, NULL);     // stride and offset: it is tightly packed
        
        // vertex colors: vbo[1] -> Attrib Array 1 -> vertexColor of the vertex shader
        glBindBuffer(GL_ARRAY_BUFFER, vbo[1]); // make it active, it is an array
        
        float* vertexColors = getStarColors(numberOfVertices);	// vertex data on the CPU
        glBufferData(GL_ARRAY_BUFFER, numberOfVertices * 3 * sizeof(GLfloat), vertexColors, GL_STATIC_DRAW);	// copy to the GPU
        
        // Map Attribute Array 1 to the current bound vertex buffer (vbo[1])
        glEnableVertexAttribArray(1);  // Vertex position
        // Data organization of Attribute Array 1
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, NULL); // Attribute Array 1, components/attribute, component type, normalize?, tightly packed
        
        delete[] vertexCoords;
        delete[] vertexColors;
    }
    
    // t: current time
    // follow: Star object to orbit around
    // pastT: time past since last call
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
            // gravitational force. a for iteration is required because the gravitational force is based on the distance between the two object, hence we need to recalculate the distance very often to get a precise value for the force.
            for (float i = 0; i <= pastT; i += val) {
                
                vec4 wD(owTx - wTx, owTy - wTy, 0, 0);
                
                float distance = sqrtf((wD.v[0] * wD.v[0]) + (wD.v[1] * wD.v[1]));
                
                vec4 normalized = wD / distance; // normalize it to get a direction vector
                
                float norm_dist = sqrtf((normalized.v[0] * normalized.v[0]) + (normalized.v[1] * normalized.v[1])); // length of the normalized direction vector
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
        
        sx = 2 + (0.5 * sinf(t * 5)); //pulsing
        sy = 2 + (0.5 * sinf(t * 5)); //pulsing
        
        rAngle = t; // Angle of the star
    }
    void Draw()
    {
        if(DOPPLER == true) //change color
        {
            glBindVertexArray(vao);		// make it active
            // vertex colors: vbo[1] -> Attrib Array 1 -> vertexColor of the vertex shader
            glBindBuffer(GL_ARRAY_BUFFER, vbo[1]); // make it active, it is an array
         
            float* vertexColors = getStarColors(numberOfVertices);	// vertex data on the CPU
            glBufferData(GL_ARRAY_BUFFER, numberOfVertices * 3 * sizeof(GLfloat), vertexColors, GL_STATIC_DRAW);	// copy to the GPU
         
            // Map Attribute Array 1 to the current bound vertex buffer (vbo[1])
            glEnableVertexAttribArray(1);  // Vertex position
            // Data organization of Attribute Array 1
            glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, NULL); // Attribute Array 1, components/attribute, component type, normalize?, tightly packed
         
            delete[] vertexColors;
        }
        
        
        mat4 M(sx,   0,  0, 0,
               0,  sy,  0, 0,
               0,   0,  0, 0,
               wTx, wTy,  0, 1); // model matrix
        
        mat4 Mrot(cos(rAngle), -sin(rAngle), 0, 0,
                  sin(rAngle), cos(rAngle),  0, 0,
                  0,           0,            1, 0,
                  0,           0,            0, 1); //rotation matrix
        
        
        mat4 MVPTransform = Mrot * M * camera.V() * camera.P(); //transform the matrix
        
        // set GPU uniform matrix variable MVP with the content of CPU variable MVPTransform
        int location = glGetUniformLocation(shaderProgram, "MVP");
        if (location >= 0) glUniformMatrix4fv(location, 1, GL_TRUE, MVPTransform); // set uniform variable MVP to the MVPTransform
        else printf("uniform MVP cannot be set\n");
        
        glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source
        
        glDrawArrays(GL_TRIANGLES, 0, numberOfVertices);	// draw a single triangle with vertices defined in vao
        
        
    }
};

// My world: three stars
Star star(vec4(0.6f, 0.2f,0.1f), 0.5);
Star star2(vec4(0.3f,0.1f,0.1f), 0.25f);
Star star3(vec4(0.4f,0,0.2f), 0.25f);

// Initialization, create an OpenGL context
void onInitialization() {
    glViewport(0, 0, windowWidth, windowHeight);
    
    // Create objects by setting up their vertex data on the GPU
    star.Create();
    star2.Create();
    star2.setPosition(-3, 0);
    star3.Create();
    star3.setPosition(3, 0);
    lineStrip.Create();
    
    // Create vertex shader from string
    unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
    if (!vertexShader) {
        printf("Error in vertex shader creation\n");
        exit(1);
    }
    glShaderSource(vertexShader, 1, &vertexSource, NULL);
    glCompileShader(vertexShader);
    checkShader(vertexShader, "Vertex shader error");
    
    // Create fragment shader from string
    unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    if (!fragmentShader) {
        printf("Error in fragment shader creation\n");
        exit(1);
    }
    glShaderSource(fragmentShader, 1, &fragmentSource, NULL);
    glCompileShader(fragmentShader);
    checkShader(fragmentShader, "Fragment shader error");
    
    // Attach shaders to a single program
    shaderProgram = glCreateProgram();
    if (!shaderProgram) {
        printf("Error in shader program creation\n");
        exit(1);
    }
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    
    // Connect Attrib Arrays to input variables of the vertex shader
    glBindAttribLocation(shaderProgram, 0, "vertexPosition"); // vertexPosition gets values from Attrib Array 0
    glBindAttribLocation(shaderProgram, 1, "vertexColor");    // vertexColor gets values from Attrib Array 1
    
    // Connect the fragmentColor to the frame buffer memory
    glBindFragDataLocation(shaderProgram, 0, "fragmentColor");	// fragmentColor goes to the frame buffer memory
    
    // program packaging
    glLinkProgram(shaderProgram);
    checkLinking(shaderProgram);
    // make this program run
    glUseProgram(shaderProgram);
}

void onExit() {
    glDeleteProgram(shaderProgram);
    printf("exit");
}

// Window has become invalid: Redraw
void onDisplay() {
    glClearColor(0.2, 0.2, 0.2, 0);							// background color
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear the screen
    
    star.Draw();
    star2.Draw();
    star3.Draw();
    lineStrip.Draw();
    
    glutSwapBuffers();									// exchange the two buffers
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
    if (key == 'd') glutPostRedisplay();         // if d, invalidate display, i.e. redraw
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {
    if(key == ' ')
    {
        camera.toggleFollow();
    }
}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) {
    if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {  // GLUT_LEFT_BUTTON / GLUT_RIGHT_BUTTON and GLUT_DOWN / GLUT_UP
        float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
        float cY = 1.0f - 2.0f * pY / windowHeight;
        lineStrip.AddPoint(cX, cY);
        glutPostRedisplay();     // redraw
    }
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
}
float lasts = 0;
// Idle event indicating that some time elapsed: do animation here
void onIdle() {
    long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program
    float sec = time / 1000.0f;				// convert msec to sec
    star.Animate(sec, nullptr, sec - lasts);// animate the star object
    star2.Animate(sec, &star, sec - lasts); // animate the smaller star object
    star3.Animate(sec, &star, sec - lasts); // animate the smaller star object
    float x,y;
    star.getPosition(&x, &y);
    camera.Animate(sec, x, y);				// animate the camera
    glutPostRedisplay();					// redraw the scene
    
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

