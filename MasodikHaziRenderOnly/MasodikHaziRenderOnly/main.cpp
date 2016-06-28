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
#include "bitmap_image.hpp"
const unsigned int windowWidth = 600, windowHeight = 600;

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Innentol modosithatod...


#ifndef M_PI
#define M_PI 3.1415926f
#endif

#define EPS 0.0001f
#define waterEPS 0.01f
#define sphereEPS 0.0001f


struct Vector {
    float x, y, z;
    
    Vector(float v = 0) : x(v), y(v), z(v) { }
    Vector(float x, float y, float z) : x(x), y(y), z(z) { }
    Vector operator+(const Vector& v) const
    {
        return Vector(x + v.x, y + v.y, z + v.z);
    }
    Vector operator-(const Vector& v) const
    {
        return Vector(x - v.x, y - v.y, z - v.z);
    }
    Vector operator*(const Vector& v) const
    {
        return Vector(x * v.x, y * v.y, z * v.z);
    }
    Vector operator/(const Vector& v) const
    {
        return Vector(x / v.x, y / v.y, z / v.z);
    }
    Vector& operator+=(const Vector& v)
    {
        x += v.x;
        y += v.y;
        z += v.z;
        return *this;
    }
    Vector operator-() const
    {
        return Vector(-x, -y, -z);
    }
    float dot(const Vector& v) const
    {
        return x*v.x + y*v.y + z*v.z;
    }
    friend float dot(const Vector& left, const Vector& right)
    {
        return left.dot(right);
    }
    Vector cross(const Vector& v) const
    {
        return Vector(y*v.z - z*v.y, z*v.x - x*v.z, x*v.y - y*v.x);
    }
    friend Vector cross(const Vector& left, const Vector& right)
    {
        return left.cross(right);
    }
    float length() const
    {
        return sqrt(x*x + y*y + z*z);
    }
    Vector normalize() const
    {
        float l = length();
        if(l > EPS) //hogy ne osszunk nullaval veletlenul se
        {
            return (*this/l);
        }
        else
        {
            return Vector();
        }
    }
    
};

class Ray {
public:
    Vector src;
    Vector dir;
    
    Ray(Vector src, Vector dir):src(src) {
        this->dir = dir.normalize();
    }
};




struct Light {
    enum Tipus { Directional, Point } type;
    Vector pos, dir;
    Vector Lout;
};

struct Material {
    Vector n, k;
    Vector kd;//kd diffuse k
    Vector ks;//ks specular k
    Vector ka;//ka ambient k
    bool isReflective;
    bool isRefractive;
    float refractiveness;
    float shininess;
    Material(Vector n, Vector k, Vector ka, Vector kd, Vector ks, bool isReflective, bool isRefractive, float shininess, float refractiveness):n(n), k(k), ka(ka), kd(kd), ks(ks), isReflective(isReflective), isRefractive(isRefractive), shininess(shininess), refractiveness(refractiveness)
    {
    }
    Vector Fresnel(Vector inDir, Vector normal)
    {
        float cos_theta = fabs(dot(normal, inDir));
        return Vector(fresnel(n.x, k.x, cos_theta), fresnel(n.y, k.y, cos_theta), fresnel(n.z, k.z, cos_theta));
    }
    
    float fresnel(float n, float k, float cos_theta) {
        float F0 = ((n-1) * (n-1) + (k * k)) / ((n + 1) * (n + 1) + (k * k));
        return F0 + (1 - F0) * pow(1-cos_theta, 5);
    }
    
    
    Vector reflect(Vector inDir, Vector normal)
    {
        Vector result = inDir - (normal * (dot(normal, inDir) * 2));
        return result;
    }
    
    ~Material() { printf("material"); }
    
    Vector shade( Vector normal, Vector viewDir, Vector lightDir, Vector inRad)
    {
        Vector reflRad(0, 0, 0);
        float cosTheta = dot(normal, lightDir);
        if(cosTheta < 0) return reflRad;
        reflRad = inRad * kd * cosTheta;;
        Vector halfway = (viewDir + lightDir).normalize();
        float cosDelta = dot(normal, halfway);
        if(cosDelta < 0) return reflRad;
        return reflRad + inRad * ks * powf(cosDelta,shininess);
    }
    
    Vector refract(Vector inDir, Vector normal) {
        float ior = refractiveness;
        float cosa = -dot(normal, inDir);
        if (cosa < 0) { cosa = -cosa; normal = -normal; ior = 1/refractiveness; }
        float disc = 1 - (1 - cosa * cosa)/ior/ior;
        if (disc < 0) return reflect(inDir, normal);
        return inDir/ior + normal * (cosa/ior - sqrt(disc));
    }
};

Material gold(Vector(0.17f, 0.35f, 1.5f), Vector(3.1f, 2.7f, 1.9f),
              Vector(0.f,0.f,0.f), Vector(0.f,0.f,0.f), Vector(0.,0.,0.), //kambient, kdiffuse, kspect
              true, false, 2.0f, 1),
silver(Vector(0.14f, 0.16f, 0.13), Vector(4.1f, 2.3f, 3.1f),
       Vector(0,0,0), Vector(0.f, 0.f, 0.f), Vector(0.f, 0.f, 0.f),
       true, false, 12.5f, 1),
glass(Vector(1.5f, 1.5f, 1.5), Vector(0.0f, 0.0f, 0.0f),
      Vector(0.2f, 0.2f, 0.2f), Vector(0.f, 0.f, 0.f), Vector(0.2f, 0.2f, 0.2f),
      true, true, 0.1f, 1.3),
grass(Vector(0.2f, 0.2f, 0.2f), Vector(0.2f, 0.2f, 0.2f),
      Vector(0.2f, 0.2f, 0.2f), Vector(0.f, 1.f, 0.f), Vector(0.f, 0.f, 0.f),
      false, false, 12.8f, 1),
poolblue(Vector(0.2f, 0.2f, 0.2f), Vector(0.2f, 0.2f, 0.2f),
         Vector(0.2f, 0.2f, 0.2f), Vector(0.2f, 0.9f, 1.f), Vector(0.225f, 0.225f, 0.225f),
         false, false, 0.1f, 1);


struct Camera {
    Vector eye;
    Vector look_at;
    Vector up;
    Vector right;
    
    Camera(Vector eye, Vector look_at, Vector up):eye(eye),look_at(look_at){
        this->up = up.normalize();
        right = ((look_at - eye).normalize().cross(this->up)).normalize();
        this->up = right.normalize().cross((look_at - eye)).normalize();
    }
    
    Ray getRay(int x, int y) {
        Vector src = look_at + right * ((2.0 * x)/windowWidth - 1) + up * ((2.0 * y)/windowHeight - 1);
        Vector dir = (src - eye).normalize();
        return Ray(src, dir);
    }
    
};
Camera camera = Camera(Vector(-13.5, 10.5, -7.5),
                       Vector(-12, 9, -7.5),
                       Vector(1, 0, 0));
/*Vector(0, 21, -13),
 Vector(0, 20.0, -13.4),
 Vector(0, 1, 0));*/
/*Vector(0, 2.2, -7.5),
 Vector(0, 2, -7.5),
 Vector(0, 0, -1));*/

struct Hit {
    float t;
    Vector position;
    Vector normal;
    Material* material;
    Hit() { t = -1; }
};


class Intersectable
{
    
public:
    Vector center;
    Material* mat;
    Intersectable(Material* mat) : mat(mat)
    {
        
    }
    
    virtual Hit intersect(Ray r) = 0;
    virtual void setCenter(Vector center){this->center = center;}
    virtual ~Intersectable()
    {
    };
};

class Triangle : public Intersectable {
public:
    Vector a, b, c, normal;
    
    Triangle(Material* mat, const Vector& a, const Vector& b, const Vector& c)
    : Intersectable(mat), a(a), b(b), c(c) {
        normal = cross((b-a).normalize(), (c-a).normalize()).normalize();
    }
    
    Hit intersect(Ray r) {
        Hit h;
        if(dot(r.dir, normal) == 0)
            return h;
        float distance = dot(a - r.src, normal) / dot(r.dir, normal);
        
        if(distance <= 0)
            return h;
        
        Vector x = r.src + (r.dir * distance);
        
        if(dot(cross(b-a, x-a), normal) > 0 && dot(cross(c-b, x-b), normal) > 0 && dot(cross(a-c, x-c), normal) > 0)
        {
            h.t = (r.src - x).length();
            h.position = x;
            h.material = this->mat;
            h.normal = normal;
        }
        
        return h;
    }
    ~Triangle()
    {
        printf("triangle\n");
    }
};
class Ellipsoid : public Intersectable
{
public:
    Vector size;
    Ellipsoid(Material* m, Vector center, Vector size) : Intersectable(m)
    {
        this->center = center;
        this->size = size;
    }
    Hit intersect(Ray ray)
    {
        Hit hit;
        Vector sizes = size*size;
        Vector rayCenter = ray.src-this->center;
        Vector origDir = ray.dir;
        Vector scaledDir = ray.dir / sizes;
        Vector scaledRayCenter = rayCenter / sizes;
        float a = origDir.dot(scaledDir);
        float b = origDir.dot(scaledRayCenter) * 2;
        float c = -1 + rayCenter.dot(scaledRayCenter);
        float diszkr = ((b * b) - (4 * a * c));
        if (fabs(diszkr) < EPS || diszkr < 0)
            return hit;
        diszkr = sqrtf(diszkr);
        float t1 = (-b+diszkr)/(2*a);
        float t2 = (-b-diszkr)/(2*a);
        if(t1<=EPS && t2<=EPS) return hit;
        float t;
        if(t1<=EPS)
            t = t2;
        else if(t2<=EPS)
            t = t1;
        else
            t=(t1<t2) ? t1 : t2;
        if(t < EPS) return hit;
        hit.t = t;
        hit.position = ray.src + origDir*t;
        hit.normal = hit.position-this->center;
        hit.normal = (hit.normal / sizes).normalize();
        hit.material = this->mat;
        return hit;
    }
    ~Ellipsoid()
    {
        
    }
};
class PoligonHalo : public Intersectable
{
public:
    Triangle* triangles[12];
    PoligonHalo(Material* material) : Intersectable(material)
    {
        
    }
    void setCenter(Vector center)
    {
        float size = 1.0f;
        float StartX = center.x-size;
        float EndX = center.x+size;
        float StartZ = center.z+size;
        float EndZ = center.z-size;
        float TopY = center.y + size;
        float BottomY = center.y-size;
        this->center = center;
        
        // also fala
        triangles[0] = (new Triangle(nullptr, Vector(StartX, BottomY, EndZ), Vector(EndX, BottomY, StartZ), Vector(StartX, BottomY, StartZ)));
        triangles[1] = (new Triangle(nullptr, Vector(EndX, BottomY, StartZ), Vector(StartX, BottomY, EndZ), Vector(EndX, BottomY, EndZ)));
        
        // bal also fala
        triangles[2] = (new Triangle(nullptr, Vector(StartX, BottomY, StartZ), center, Vector(StartX, BottomY, EndZ)));
        // jobb also fala
        triangles[3] = (new Triangle(nullptr, Vector(EndX, BottomY, StartZ), Vector(EndX, BottomY, EndZ), center));
        // szembe also fala
        triangles[4] = (new Triangle(nullptr, Vector(StartX, BottomY, StartZ), Vector(EndX, BottomY, StartZ), center));
        // hatso also fala
        triangles[5] = (new Triangle(nullptr, Vector(StartX, BottomY, EndZ), center, Vector(EndX, BottomY, EndZ)));
        
        // also fala
        triangles[6] = (new Triangle(nullptr, Vector(StartX, TopY, EndZ), Vector(StartX, TopY, StartZ), Vector(EndX, TopY, StartZ)));
        triangles[7] = (new Triangle(nullptr, Vector(EndX, TopY, StartZ), Vector(EndX, TopY, EndZ), Vector(StartX, TopY, EndZ)));
        
        // bal also fala
        triangles[8] = (new Triangle(nullptr, Vector(StartX, TopY, StartZ), Vector(StartX, TopY, EndZ), center));
        // jobb also fala
        triangles[9] = (new Triangle(nullptr, Vector(EndX, TopY, StartZ), center, Vector(EndX, TopY, EndZ)));
        // szembe also fala
        triangles[10] = (new Triangle(nullptr, Vector(StartX, TopY, StartZ), center, Vector(EndX, TopY, StartZ)));
        // hatso also fala
        triangles[11] = (new Triangle(nullptr, Vector(StartX, TopY, EndZ), Vector(EndX, TopY, EndZ), center));
    }
    Hit intersect(Ray ray)
    {
        Hit minHit;
        
        minHit.t = 1001;
        for (int i = 0; i < 12; i++) {
            Hit trHit = triangles[i]->intersect(ray);
            if(trHit.t > 0)
            {
                if(minHit.t > trHit.t)
                    minHit = trHit;
            }
        }
        if(minHit.t < 1000)
        {
            minHit.material = this->mat;
            return minHit;
        }
        return Hit();
    }
    ~PoligonHalo()
    {
        printf("poligonhalo\n");
        for (int i = 0; i < 12; i++) {
            delete triangles[i];
        }
    }
};

class Bolya : public Intersectable
{
public:
    Intersectable* topPart;
    
    Triangle* triangles[12];
    Bolya(Material* material, Intersectable* topPart) : Intersectable(material), topPart(topPart)
    {
        
    }
    void setCenter(Vector center)
    {
        float StartX = center.x-1.5f;
        float EndX = center.x+1.5f;
        float StartZ = center.z+1.5f;
        float EndZ = center.z-1.5f;
        float TopY = center.y + 1.5f;
        float BottomY = center.y-1.5f;
        this->center = center;
        Vector nCenter = center;
        nCenter.y = center.y+1.5f+1.f;
        topPart->setCenter(nCenter);
        
        //medence also fala
        triangles[0] = (new Triangle(nullptr, Vector(StartX, BottomY, EndZ), Vector(EndX, BottomY, StartZ), Vector(StartX, BottomY, StartZ)));
        triangles[1] = (new Triangle(nullptr, Vector(EndX, BottomY, StartZ), Vector(StartX, BottomY, EndZ), Vector(EndX, BottomY, EndZ)));
        
        //medence jobb fala
        triangles[2] = (new Triangle(nullptr, Vector(EndX, BottomY, StartZ), Vector(EndX, BottomY, EndZ), Vector(EndX, TopY, EndZ)));
        triangles[3] = (new Triangle(nullptr, Vector(EndX, BottomY, StartZ), Vector(EndX, TopY, EndZ), Vector(EndX, TopY, StartZ)));
        
        //medence hatso fala
        triangles[4] = (new Triangle(nullptr, Vector(StartX, BottomY, EndZ), Vector(EndX, TopY, EndZ), Vector(EndX, BottomY, EndZ)));
        triangles[5] = (new Triangle(nullptr, Vector(EndX, TopY, EndZ), Vector(StartX, BottomY, EndZ), Vector(StartX, TopY, EndZ)));
        
        //medence bal fala
        triangles[6] = (new Triangle(nullptr, Vector(StartX, BottomY, StartZ), Vector(StartX, TopY, EndZ), Vector(StartX, BottomY, EndZ)));
        triangles[7] = (new Triangle(nullptr, Vector(StartX, BottomY, StartZ), Vector(StartX, TopY, StartZ), Vector(StartX, TopY, EndZ)));
        
        //medence elso fala
        triangles[8] = (new Triangle(nullptr, Vector(StartX, BottomY, StartZ), Vector(EndX, BottomY, StartZ), Vector(EndX, TopY, StartZ)));
        triangles[9] = (new Triangle(nullptr, Vector(EndX, TopY, StartZ), Vector(StartX, TopY, StartZ), Vector(StartX, BottomY, StartZ)));
        
        //medence felso fala
        triangles[10] = (new Triangle(nullptr, Vector(StartX, TopY, EndZ), Vector(StartX, TopY, StartZ), Vector(EndX, TopY, StartZ)));
        triangles[11] = (new Triangle(nullptr, Vector(EndX, TopY, StartZ), Vector(EndX, TopY, EndZ), Vector(StartX, TopY, EndZ)));
    }
    Hit intersect(Ray ray)
    {
        Hit minHit;
        
        minHit.t = 1001;
        for (int i = 0; i < 12; i++) {
            Hit trHit = triangles[i]->intersect(ray);
            if(trHit.t > 0)
            {
                if(minHit.t > trHit.t)
                    minHit = trHit;
            }
        }
        Hit tpHit = topPart->intersect(ray);
        tpHit.material = this->mat;
        if(tpHit.t > 0 && minHit.t > tpHit.t)
            minHit = tpHit;
        if(minHit.t < 1000)
        {
            minHit.material = this->mat;
            return minHit;
        }
        return Hit();
    }
    
    ~Bolya()
    {
        printf("bolya\n");
        for (int i = 0; i < 12; i++) {
            delete triangles[i];
        }
        delete topPart;
    }
    
};


float t = 1.4f; //1 sec = 3.14f
class Wave : public Intersectable
{
public:
    Triangle *f1A, *f2A, *f1F, *f2F;
    float centerHeight;
    Vector p1, p2;
    float amplitude1, frequency1;
    float amplitude2, frequency2;
    float x,y,width, height;
    float cx,cy,cwidth, cheight;
    float maxy, miny;
    Wave(Material* m, Vector xy, Vector wh, Vector rxy, Vector rwh, float centerHeight, Vector p1, Vector p2, float amplitude1, float amplitude2, Bolya* bolya1, Bolya* bolya2): Intersectable(m)
    {
        this->x = rxy.x;
        this->y = rxy.y;
        this->width = rwh.x;
        this->height = rwh.y;
        this->cx = xy.x;
        this->cy = xy.y;
        this->cwidth = wh.x;
        this->cheight = wh.y;
        maxy = centerHeight + amplitude1 + amplitude2 + 0.1f;
        miny = centerHeight - (amplitude1 + amplitude2 + 0.1f);
        Triangle* F1 = new Triangle(nullptr, Vector(x, maxy, y-height), Vector(x, maxy, y), Vector(x+width, maxy, y));
        Triangle* F2 = new Triangle(nullptr, Vector(x+width, maxy, y), Vector(x+width, maxy, y-height), Vector(x, maxy, y-height));
        
        Triangle* A1 = new Triangle(nullptr, Vector(x, miny, y-height), Vector(x, miny, y), Vector(x+width, miny, y));
        Triangle* A2 = new Triangle(nullptr, Vector(x+width, miny, y), Vector(x+width, miny, y-height), Vector(x, miny, y-height));
        this->f1A = A1;
        this->f2A = A2;
        this->f1F = F1;
        this->f2F = F2;
        this->centerHeight = centerHeight;
        this->p1 = p1;
        this->p2 = p2;
        this->amplitude1 = amplitude1;
        this->frequency1 = M_PI;
        this->amplitude2 = amplitude2;
        this->frequency2 = M_PI;
        Vector bolyaCenter = p1;
        bolyaCenter.y = getHeightByPos(p1.x, p1.z);
        
        bolya1->setCenter(bolyaCenter);
        
        Vector bolyaCenter2 = p2;
        bolyaCenter2.y = getHeightByPos(p2.x, p2.z);
        bolya2->setCenter(bolyaCenter2);
        
    }
    
    float getHeightByPos(float x, float z)
    {
        Vector h = Vector(x, centerHeight, z);
        float distanceP1 = (h - p1).length();
        float distanceP2 = (h - p2).length();
        
        float yP1 = amplitude1 * cosf((frequency1 * distanceP1) + t);
        float yP2 = amplitude2 * cosf((frequency2 * distanceP2) + t);
        
        return centerHeight + (yP1 + yP2);
    }
    
    Vector getNormalByPos(float x, float z)
    {
        Vector h = Vector(x, centerHeight, z);
        Vector fromP1 = (h - p1);
        Vector fromP2 = (h - p2);
        float x1 = fromP1.x;
        float z1 = fromP1.z;
        float x2 = fromP2.x;
        float z2 = fromP2.z;
        float base1 = sinf((frequency1 * sqrtf((x1*x1) + (z1*z1))) + t) * powf(((x1*x1) + (z1*z1)), -0.5f);
        float base2 = sinf((frequency2 * sqrtf((x2*x2) + (z2*z2))) + t) * powf(((x2*x2) + (z2*z2)), -0.5f);
        float A1 = -1 * amplitude1 * base1 * 0.5f * frequency1 * 2 * x1;
        A1 = A1 + (-1 * amplitude2 * base2 * 0.5f * frequency2 * 2 * x2);
        
        float B1 = -1 * amplitude1 * base1 * 0.5f * frequency1 * 2 * z1;
        B1 = B1 + (-1 * amplitude2 * base2 * 0.5f * frequency2 * 2 * z2);
        Vector ret = cross(Vector(1, A1, 0), Vector(0, -B1, 1));
        return ret;
    }
    
    Hit intersect(Ray ray)
    {
        Hit hit;
        
        
        Hit h1A = f1A->intersect(ray);
        Hit h2A = f2A->intersect(ray);
        Hit planehitA;
        if(h1A.t >= 0)
            planehitA = h1A;
        else if(h2A.t >= 0)
            planehitA = h2A;
        else
            return hit;
        
        Hit h1F = f1F->intersect(ray);
        Hit h2F = f2F->intersect(ray);
        Hit planehitF;
        if(h1F.t >= 0)
            planehitF = h1F;
        else if(h2F.t >= 0)
            planehitF = h2F;
        else
        {
            float rayPosHeight = getHeightByPos(ray.src.x, ray.src.z);
            if(ray.src.y < maxy && rayPosHeight+0.05 < ray.src.y)
            {
                
                ray.dir = ray.dir * -1;
                h1F = f1F->intersect(ray);
                h2F = f2F->intersect(ray);
                if(h1F.t >= 0)
                    planehitF = h1F;
                else if(h2F.t >= 0)
                    planehitF = h2F;
            }
            else
            {
                return hit;
            }
        }
        
        if(!isInRectangle(planehitA.position) && !isInRectangle(planehitF.position))
            return hit;
        
        Vector dir = (planehitA.position - planehitF.position).normalize() * waterEPS * 0.9f;
        float heightDiff = 999;
        Vector linePos = planehitF.position;
        while (!(heightDiff > -waterEPS && heightDiff < +waterEPS)) {
            float lineHeight = linePos.y;
            float waveHeight = getHeightByPos(linePos.x, linePos.z);
            heightDiff = (lineHeight - waveHeight);
            linePos = linePos + dir;
            
            if(linePos.y > maxy || linePos.y < miny)
                break;
        }
        if(!isInRectangle(linePos))
            return hit;
        hit.position = linePos;
        hit.material = this->mat;
        hit.normal = getNormalByPos(linePos.x, linePos.z).normalize();
        hit.t = (ray.src - linePos).length();
        return hit;
    }
    
    bool isInRectangle(Vector pos)
    {
        return (pos.x >= cx && pos.x <= cx + cwidth &&
                pos.z <= cy && pos.z >= cy - cheight);
    }
    ~Wave()
    {
        printf("wave\n");
        delete f1A;
        delete f1F;
        delete f2A;
        delete f2F;
    }
    
};
static int renderCount = 0;
struct Scene {
    const int maxdepth = 3;
    const Vector La = Vector(0.52f, 0.81f, 0.909f);
    Intersectable* ins[100];
    Light lights[3];
    size_t numberOfLights;
    size_t numberOfObjects;
    
    Scene() {
        numberOfLights = 0;
        numberOfObjects = 0;
    }
    void AddObject(Intersectable *o) {
        ins[numberOfObjects++] = o;
    }
    void AddLight(const Light& l) {
        lights[numberOfLights++] = l;
    }
    void build()
    {
        Vector sunDir = Vector(0, 2.24603677390422f, -1).normalize();
        Light l1 = {Light::Directional, Vector(), sunDir, Vector(.3f, .3f, .3f)};
        //Light l2 = {Light::Point, Vector(0,10, -10), Vector(), Vector(10.f, 10.f, 10.f)};
        Vector sunPos = Vector(-1,2.24603677390422f,0.f).normalize() * 149597870700;
        //Light l2 = {Light::Ambient, sunPos, Vector(), Vector(110000000000.f, 110000000000.f, 110000000000.f)};
        this->AddLight(l1);
        //this->AddLight(l2);
        
        
        float poolStartX = -5.0f;
        float poolEndX = 5.0f;
        float poolStartZ = -5.0f;
        float poolEndZ = -55.0f;
        float poolTopY = 0.0f;
        float poolBottomY = -3.0f;
        
        float grassStartX = poolStartX - 15.0f;
        float grassEndX = poolEndX + 15.0f;
        float grassStartZ = poolStartZ + 10.0f;
        float grassEndZ = poolStartZ;
        this->AddObject(new Triangle(&grass, Vector(grassStartX, 0, grassEndZ), Vector(grassStartX, 0, grassStartZ), Vector(grassEndX, 0,grassStartZ)));
        this->AddObject(new Triangle(&grass, Vector(grassEndX, 0, grassStartZ), Vector(grassEndX, 0, grassEndZ), Vector(grassStartX, 0, grassEndZ)));
        
        grassStartX = poolStartX - 40.0f;
        grassEndX = poolStartX;
        grassStartZ = poolStartZ;
        grassEndZ = poolEndZ;
        
        this->AddObject(new Triangle(&grass, Vector(grassStartX, 0, grassEndZ), Vector(grassStartX, 0, grassStartZ), Vector(grassEndX, 0,grassStartZ)));
        this->AddObject(new Triangle(&grass, Vector(grassEndX, 0, grassStartZ), Vector(grassEndX, 0, grassEndZ), Vector(grassStartX, 0, grassEndZ)));
        
        grassStartX = poolEndX;
        grassEndX = poolEndX + 40.0f;
        grassStartZ = poolStartZ;
        grassEndZ = poolEndZ;
        
        this->AddObject(new Triangle(&grass, Vector(grassStartX, 0, grassEndZ), Vector(grassStartX, 0, grassStartZ), Vector(grassEndX, 0,grassStartZ)));
        this->AddObject(new Triangle(&grass, Vector(grassEndX, 0, grassStartZ), Vector(grassEndX, 0, grassEndZ), Vector(grassStartX, 0, grassEndZ)));
        
        grassStartX = poolStartX - 30.0f;
        grassEndX = poolEndX + 30.0f;
        grassStartZ = poolEndZ;
        grassEndZ = poolEndZ - 10.0f;
        this->AddObject(new Triangle(&grass, Vector(grassStartX, 0, grassEndZ), Vector(grassStartX, 0, grassStartZ), Vector(grassEndX, 0,grassStartZ)));
        this->AddObject(new Triangle(&grass, Vector(grassEndX, 0, grassStartZ), Vector(grassEndX, 0, grassEndZ), Vector(grassStartX, 0, grassEndZ)));
        
        //medence also fala
        this->AddObject(new Triangle(&poolblue, Vector(poolStartX, poolBottomY, poolEndZ), Vector(poolStartX, poolBottomY, poolStartZ), Vector(poolEndX, poolBottomY, poolStartZ)));
        this->AddObject(new Triangle(&poolblue, Vector(poolEndX, poolBottomY, poolStartZ), Vector(poolEndX, poolBottomY, poolEndZ), Vector(poolStartX, poolBottomY, poolEndZ)));
        
        //medence jobb fala
        this->AddObject(new Triangle(&poolblue, Vector(poolEndX, poolBottomY, poolStartZ), Vector(poolEndX, poolTopY, poolEndZ), Vector(poolEndX, poolBottomY, poolEndZ)));
        this->AddObject(new Triangle(&poolblue, Vector(poolEndX, poolBottomY, poolStartZ), Vector(poolEndX, poolTopY, poolStartZ), Vector(poolEndX, poolTopY, poolEndZ)));
        
        //medence hatso fala
        this->AddObject(new Triangle(&poolblue, Vector(poolStartX, poolBottomY, poolEndZ), Vector(poolEndX, poolBottomY, poolEndZ), Vector(poolEndX, poolTopY, poolEndZ)));
        this->AddObject(new Triangle(&poolblue, Vector(poolEndX, poolTopY, poolEndZ), Vector(poolStartX, poolTopY, poolEndZ), Vector(poolStartX, poolBottomY, poolEndZ)));
        
        //medence bal fala
        this->AddObject(new Triangle(&poolblue, Vector(poolStartX, poolBottomY, poolStartZ), Vector(poolStartX, poolBottomY, poolEndZ), Vector(poolStartX, poolTopY, poolEndZ)));
        this->AddObject(new Triangle(&poolblue, Vector(poolStartX, poolBottomY, poolStartZ), Vector(poolStartX, poolTopY, poolEndZ), Vector(poolStartX, poolTopY, poolStartZ)));
        
        //medence elso fala
        this->AddObject(new Triangle(&poolblue, Vector(poolStartX, poolBottomY, poolStartZ), Vector(poolEndX, poolTopY, poolStartZ), Vector(poolEndX, poolBottomY, poolStartZ)));
        this->AddObject(new Triangle(&poolblue, Vector(poolEndX, poolTopY, poolStartZ), Vector(poolStartX, poolBottomY, poolStartZ), Vector(poolStartX, poolTopY, poolStartZ)));
        
        
        Ellipsoid* ellip = new Ellipsoid(&grass, Vector(), Vector(0.8f, 2.f, 0.8f));
        PoligonHalo* ph = new PoligonHalo(&poolblue);
        Bolya* bolya = new Bolya(&gold, ellip);
        Bolya* bolya2 = new Bolya(&silver, ph);
        
        
        this->AddObject(bolya);
        this->AddObject(bolya2);
        float centerHeight = -0.5f;
        this->AddObject(new Wave(&glass, Vector(poolStartX, poolStartZ+1, 0), Vector(poolEndX - poolStartX, fabs(poolEndZ - poolStartZ) + 2, 0), Vector(poolStartX - 10, poolStartZ + 10, 0), Vector(90, 90, 0), centerHeight, Vector(0, centerHeight, poolStartZ - 10.0f), Vector(0, centerHeight, poolEndZ + 10.0f), 0.3f, 0.3f, bolya2, bolya));
        
        
        
    }
    
    void render(int count)
    {
        Vector image[windowHeight * windowWidth];
        
        bitmap_image outImage(600,600);
        for(int y = 0; y < windowHeight; y++)
        {
            for(int x = 0; x < windowWidth; x++)
            {
                Ray ray = camera.getRay(x, y);
                Vector c = trace(ray, 0);
                Vector text(c.x, c.y, c.z);
                image[(y*windowHeight) + x] = text;
                int r = (int)(c.x*255);
                if(r > 255)
                    r = 255;
                int g = (int)(c.y*255);
                if(g > 255)
                    g = 255;
                int b = (int)(c.z*255);
                if(b > 255)
                    b = 255;
                outImage.set_pixel(x, windowHeight - 1 - y, (char)r, (char)g, (char)b);
            }
        }
        std::string str = std::to_string(count);
        std::string s = "render" + str + ".bmp";
        
        outImage.save_image(s);
        
    }
    
    Hit firstIntersect(Ray ray) {
        Hit bestHit;
        
        for (int i = 0; i < numberOfObjects; i++) {
            Hit hit = ins[i]->intersect(ray); //  hit.t < 0 if no intersection
            if(hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t)) 	bestHit = hit;
        }
        
        return bestHit;
    }
    
    
    Vector trace(Ray ray, int depth) {
        if (depth > maxdepth) return La;
        
        Hit hit = firstIntersect(ray);
        if(hit.t < 0) return La; // nothing
        Vector outRadiance = hit.material->ka * La;
        float eps = hit.normal.dot(ray.dir) < 0 ? EPS:-EPS;
        Vector neps = hit.normal.normalize() * eps;
        for(int i = 0; i < numberOfLights; i++)
        {
            Light l = lights[i];
            Vector backDir = l.pos - hit.position;
            float length = (backDir).length();
            if(l.type == Light::Directional)
            {
                backDir = l.dir;
                length = 1;
            }
            Ray shadowRay(hit.position + neps, backDir);
            Hit shadowHit = firstIntersect(shadowRay);
            
            float multiplier;
            if(shadowHit.t < 0 || (shadowHit.t > length && l.type != Light::Directional))
            {
                multiplier = 1.f;
            }
            else
            {
                if(shadowHit.material->isRefractive)
                    multiplier = 0.9f;
                else
                    multiplier = 0.f;
                
            }
            outRadiance += hit.material->shade(hit.normal, ray.dir, backDir, (l.Lout * multiplier * (1.0f/(length * length))));
        }
        
        if(hit.material->isReflective){
            Vector reflectionDir = hit.material->reflect(ray.dir,hit.normal);
            Ray reflectedRay(hit.position + neps, reflectionDir);
            Vector sunDir = Vector(0, 2.24603677390422f, -1).normalize();
            float degree = acosf(dot(reflectionDir.normalize(), sunDir)) * (180/M_PI);
            if(fabs(degree) < 3.56f) //0.56 kellene, de akkor nem latvanyos, mert csak 1-2 pont
            {
                outRadiance += lights[0].Lout;
            }
            
            outRadiance += trace(reflectedRay,depth+1)* hit.material->Fresnel(ray.dir, hit.normal);
        }
        if(hit.material->isRefractive) {
            Vector refractionDir = hit.material->refract(ray.dir,hit.normal);
            Ray refractedRay(hit.position - neps, refractionDir);
            outRadiance += trace(refractedRay,depth+1)*(Vector(1,1,1)-hit.material->Fresnel(ray.dir, hit.normal));
        }
        return outRadiance;
    }
    ~Scene() {
        printf("scene\n");
        for(int i = 0; i < numberOfObjects; ++i) {
            delete ins[i];
        }
    }
    
};

int main(int argc, char * argv[]) {
    float cameraZ = 0;
    int counter = 0;
    t = 0;
    /*while(t <= (3.14f * 5))
    {
        Scene* scene = new Scene();
        camera = Camera(Vector(-13.5, 10.5, -15),
                        Vector(-12, 9, -15),
                        Vector(1, 0, 0));
        scene->build();
        scene->render(counter++);
        delete scene;
        t = t + 0.1f;
        printf("renderDone: %d", renderCount);
    }*/
    /*t = 15.7f;
     counter = 158;
    while(cameraZ <= 30)
    {
        Scene* scene = new Scene();
        camera = Camera(Vector(-13.5, 10.5, -15 - cameraZ),
                    Vector(-12, 9, -15 - cameraZ),
                    Vector(1, 0, 0));
        scene->build();
        scene->render(counter++);
        delete scene;
        t = t + 0.1f;
        cameraZ = cameraZ + 0.2f;
        printf("renderDone: %d", renderCount);
    }*/
    
    counter = 308;
    t = 30.7f;
    cameraZ = 30.2f;
    float currT = t;
    while(t - currT <= (3.14f * 5))
    {
        Scene* scene = new Scene();
        camera = Camera(Vector(-13.5, 10.5, -15 - cameraZ),
                        Vector(-12, 9, -15 - cameraZ),
                        Vector(1, 0, 0));
        scene->build();
        scene->render(counter++);
        delete scene;
        t = t + 0.1f;
        printf("renderDone: %d", renderCount);
    }
    return 1;
}