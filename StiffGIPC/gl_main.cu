//
// gl_main.cpp
// GIPC
//
// created by Kemeng Huang on 2022/12/01
// Copyright (c) 2024 Kemeng Huang. All rights reserved.
//

#include "GL/glew.h"
#include <agipc/agipc_criterion.cuh>
#include "GL/freeglut.h"
#include <fstream>
#include <iostream>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <limits>
#include <cuda_runtime.h>
#include <map>
// #include "GIPC.cuh"
#include "device_launch_parameters.h"
#include "mlbvh.cuh"
#include <stdio.h>
#include "load_mesh.h"
#include "cuda_tools/cuda_tools.h"
#include <queue>
//#include "timer.h"
#include "femEnergy.cuh"
#include "gpu_eigen_libs.cuh"
#include "fem_parameters.h"
#include "gipc_path.h"
#include <gipc/type_define.h>
#include <filesystem>
#include <gipc/statistics.h>
#include <gipc/runtime_options.h>
#include <linear_system/utils/spmv.h>
#include <gipc/utils/simple_scene_importer.h>
#include <Eigen/Geometry>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/device_ptr.h>
#include <GIPC.cuh>

auto             assets_dir = std::string{gipc::assets_dir()};
std::string      metis_dir  = assets_dir + "sorted_mesh/";
double           collision_detection_buff_scale = 1;
double           motion_rate                    = 1;
double           linear_system_buff_scale       = 1.0;
mesh_obj         obj;
lbvh_f           bvh_f;
lbvh_e           bvh_e;
GIPC             ipc;
device_TetraData d_tetMesh;
tetrahedra_obj   tetMesh;
std::vector<Node>     nodes;
std::vector<AABB>     bvs;
std::vector<std::string>   obj_pathes;
int              initPath = 0;
int   step      = 0;
int   frameId   = 0;
int   surfNumId = 0;
float xRot      = 0.0f;
float yRot      = 0.f;
float xTrans    = 0;
float yTrans    = 0;
float zTrans    = 0;
int   ox;
int   oy;
int   buttonState;
float xRotLength    = 0.0f;
float yRotLength    = 0.0f;
float window_width  = 1000;
float window_height = 1000;
int   s_dimention   = 3;
bool  saveSurface   = false;
bool  change        = false;
bool  screenshot    = false;

bool isSetShader = false;

bool drawbvh     = false;
bool drawSurface = true;

bool stop = true;

double3 center;
double3 Ssize;

GLuint PN_vbo_;
GLuint VAO;
GLuint color_vbo_;
//GLuint color_vao_;
GLuint normal_vbo_;
//GLuint normal_vao_;
GLuint v;
GLuint f;
GLuint shaderProgram;

int            clothFaceOffset = 0;
int            bodyVertOffset  = 0;
double         global_offset   = 1.0;
std::vector<std::string> files;
std::vector<int>    file_vert_offsets;
std::vector<int>    file_tet_offsets;
gipc::RuntimeOptions runtime_options;
int figure12_cloth_vertex_count   = 0;
int figure12_cloth_triangle_count = 0;
int figure15_cloth_vertex_count   = 0;
int figure15_cloth_triangle_count = 0;
constexpr double figure12_primary_bunny_young_modulus   = 1e7;
constexpr double figure12_secondary_bunny_young_modulus = 1e4;

void Init_CUDA()
{
    cudaError_t cudaStatus = cudaSetDevice(0);
    if(cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        exit(0);
    }
}

#pragma pack(push, 1)
typedef struct
{
    uint16_t bfType;
    uint32_t bfSize;
    uint16_t bfReserved1;
    uint16_t bfReserved2;
    uint32_t bfOffBits;
} mBITMAPFILEHEADER;
#pragma pack(pop)

#pragma pack(push, 1)
typedef struct
{
    uint32_t biSize;
    int32_t  biWidth;
    int32_t  biHeight;
    uint16_t biPlanes;
    uint16_t biBitCount;
    uint32_t biCompression;
    uint32_t biSizeImage;
    int32_t  biXPelsPerMeter;
    int32_t  biYPelsPerMeter;
    uint32_t biClrUsed;
    uint32_t biClrImportant;
} mBITMAPINFOHEADER;
#pragma pack(pop)

bool WriteBitmapFile(int width, int height, const std::string& file_name, unsigned char* bitmapData)
{
    mBITMAPFILEHEADER bitmapFileHeader;
    memset(&bitmapFileHeader, 0, sizeof(mBITMAPFILEHEADER));
    bitmapFileHeader.bfSize = sizeof(mBITMAPFILEHEADER);
    bitmapFileHeader.bfType = 0x4d42;  //BM
    bitmapFileHeader.bfOffBits = sizeof(mBITMAPFILEHEADER) + sizeof(mBITMAPINFOHEADER);

    mBITMAPINFOHEADER bitmapInfoHeader;
    memset(&bitmapInfoHeader, 0, sizeof(mBITMAPINFOHEADER));
    bitmapInfoHeader.biSize        = sizeof(mBITMAPINFOHEADER);
    bitmapInfoHeader.biWidth       = width;
    bitmapInfoHeader.biHeight      = height;
    bitmapInfoHeader.biPlanes      = 1;
    bitmapInfoHeader.biBitCount    = 24;
    bitmapInfoHeader.biCompression = 0L;
    bitmapInfoHeader.biSizeImage   = width * std::abs(height) * 3;

    //////////////////////////////////////////////////////////////////////////
    FILE*         filePtr;
    unsigned char tempRGB;
    int           imageIdx;

    for(imageIdx = 0; imageIdx < (int)bitmapInfoHeader.biSizeImage; imageIdx += 3)
    {
        tempRGB                  = bitmapData[imageIdx];
        bitmapData[imageIdx]     = bitmapData[imageIdx + 2];
        bitmapData[imageIdx + 2] = tempRGB;
    }

    filePtr = fopen(file_name.c_str(), "wb");
    if(NULL == filePtr)
    {
        return false;
    }

    fwrite(&bitmapFileHeader, sizeof(mBITMAPFILEHEADER), 1, filePtr);

    fwrite(&bitmapInfoHeader, sizeof(mBITMAPINFOHEADER), 1, filePtr);

    fwrite(bitmapData, bitmapInfoHeader.biSizeImage, 1, filePtr);

    fclose(filePtr);
    return true;
}

void SaveScreenShot(int width, int height, const std::string& file_name)
{
    int   data_len    = height * width * 3;  // bytes
    void* screen_data = malloc(data_len);
    memset(screen_data, 0, data_len);
    glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, screen_data);
    WriteBitmapFile(width, height, file_name + ".bmp", (unsigned char*)screen_data);
    free(screen_data);
}

void saveSurfaceMesh(const std::string& path)
{
    std::stringstream ss;
    ss << path;
    ss.fill('0');
    ss.width(5);
    ss << (surfNumId++) / 1;  // / 10;
    //if (surfNumId % 10 != 0) return;
    ss << ".obj";
    std::string file_path = ss.str();
    std::ofstream    outSurf(file_path);

    std::map<int, int> meshToSurf;
    outSurf << "s 1" << std::endl;
    for(int i = 0; i < tetMesh.surfVerts.size(); i++)
    {
        const auto& pos = tetMesh.vertexes[tetMesh.surfVerts[i]];
        outSurf << "v " << pos.x << " " << pos.y << " " << pos.z << std::endl;
        meshToSurf[tetMesh.surfVerts[i]] = i;
    }

    for(int i = 0; i < tetMesh.surface.size(); i++)
    {
        const auto& tri = tetMesh.surface[i];
        outSurf << "f " << meshToSurf[tri.x] + 1 << " " << meshToSurf[tri.y] + 1
                << " " << meshToSurf[tri.z] + 1 << std::endl;
    }
    outSurf.close();
}


void saveTets(const std::string& path)
{
    int tetIdoffset = 0;
    for(int ii = 0; ii < 4096; ii++)
    {
        //tetMesh.output_tetrahedraMesh
        std::stringstream ss;
        ss << path;
        ss << ii;  // / 10;
        //if (surfNumId % 10 != 0) return;
        ss << ".msh";
        std::string file_path = ss.str();
        std::ofstream    outmsh1(file_path);

        std::map<int, int> meshToSurf;
        //outSurf << "s 1" << std::endl;
        outmsh1 << "$Nodes\n";
        outmsh1 << file_vert_offsets[ii + 1] - file_vert_offsets[ii] << std::endl;
        for(int i = 0; i < file_vert_offsets[ii + 1] - file_vert_offsets[ii]; i++)
        {
            const auto& pos = tetMesh.vertexes[i + file_vert_offsets[ii]];
            outmsh1 << i + 1 << " " << pos.x << " " << pos.y << " " << pos.z << std::endl;
            meshToSurf[i + file_vert_offsets[ii]] = i;
        }
        outmsh1 << "$Elements\n";
        outmsh1 << file_tet_offsets[ii + 1] << std::endl;

        for(int i = 0; i < file_tet_offsets[ii + 1]; i++)
        {
            int tetId = i + tetIdoffset;
            outmsh1 << i + 1 << " 4 0 " << meshToSurf[tetMesh.tetrahedras[tetId].x] + 1
                    << " " << meshToSurf[tetMesh.tetrahedras[tetId].y] + 1
                    << " " << meshToSurf[tetMesh.tetrahedras[tetId].z] + 1 << " "
                    << meshToSurf[tetMesh.tetrahedras[tetId].w] + 1 << std::endl;
        }
        tetIdoffset += file_tet_offsets[ii + 1];
        outmsh1.close();
    }
}

void draw_box2D(float ox, float oy, float width, float height)
{
    glLineWidth(2.5f);
    glColor3f(0.8f, 0.8f, 0.8f);

    glBegin(GL_LINES);

    glVertex3f(ox, oy, 0);
    glVertex3f(ox + width, oy, 0);

    glVertex3f(ox, oy, 0);
    glVertex3f(ox, oy + height, 0);

    glVertex3f(ox + width, oy, 0);
    glVertex3f(ox + width, oy + height, 0);

    glVertex3f(ox + width, oy + height, 0);
    glVertex3f(ox, oy + height, 0);

    glEnd();
}

void draw_box3D(float ox, float oy, float oz, float width, float height, float length, int boxType = 0)
{
    glLineWidth(0.5f);
    glColor3f(0.8f, 0.8f, 0.1f);
    if(boxType == 1)
    {
        glLineWidth(1.5f);
        glColor3f(0.8f, 0.8f, 0.8f);
    }
    glBegin(GL_LINES);

    glVertex3f(ox, oy, oz);
    glVertex3f(ox + width, oy, oz);

    glVertex3f(ox, oy, oz);
    glVertex3f(ox, oy + height, oz);

    glVertex3f(ox, oy, oz);
    glVertex3f(ox, oy, oz + length);

    glVertex3f(ox + width, oy, oz);
    glVertex3f(ox + width, oy + height, oz);

    glVertex3f(ox + width, oy + height, oz);
    glVertex3f(ox, oy + height, oz);

    glVertex3f(ox, oy + height, oz + length);
    glVertex3f(ox, oy, oz + length);

    glVertex3f(ox, oy + height, oz + length);
    glVertex3f(ox, oy + height, oz);

    glVertex3f(ox + width, oy, oz);
    glVertex3f(ox + width, oy, oz + length);

    glVertex3f(ox, oy, oz + length);
    glVertex3f(ox + width, oy, oz + length);

    glVertex3f(ox + width, oy + height, oz);
    glVertex3f(ox + width, oy + height, oz + length);

    glVertex3f(ox + width, oy + height, oz + length);
    glVertex3f(ox + width, oy, oz + length);

    glVertex3f(ox, oy + height, oz + length);
    glVertex3f(ox + width, oy + height, oz + length);

    glEnd();
}

void draw_lines(float ox, float oy, float oz, float width, float height, float length)
{
    glLineWidth(0.5f);
    glColor3f(0.8f, 0.8f, 0.8f);

    glBegin(GL_LINES);
    int numbers = 20;
    for(int i = 0; i <= numbers; i++)
    {
        //glVertex3f(ox, oy, oz);
        glVertex3f(ox + width * i / numbers, oy, 0);
        glVertex3f(ox + width * i / numbers, oy + height, 0);
    }

    for(int i = 0; i <= numbers; i++)
    {
        //glVertex3f(ox, oy, oz);
        glVertex3f(ox, oy + height * i / numbers, 0);
        glVertex3f(ox + width, oy + height * i / numbers, 0);
    }

    glEnd();


    glLineWidth(1.5f);
    glColor3f(0.8f, 0.8f, 0.f);
    glBegin(GL_LINES);
    glVertex3f(ox + width / 2, oy, 0);
    glVertex3f(ox + width / 2, oy + height, 0);

    glVertex3f(ox, oy + height / 2, 0);
    glVertex3f(ox + width, oy + height / 2, 0);

    glEnd();
}

void draw_mesh3D()
{
    glEnable(GL_DEPTH_TEST);
    glLineWidth(1.5f);
    glColor3f(0.9f, 0.1f, 0.1f);
    const std::vector<uint3>& surf = tetMesh.surface;  //obj.faces;
    glBegin(GL_TRIANGLES);


    for(int j = 0; j < tetMesh.surface.size(); j++)
    {
        glVertex3f((tetMesh.vertexes[surf[j].x].x),
                   (tetMesh.vertexes[surf[j].x].y),
                   (tetMesh.vertexes[surf[j].x].z));
        glVertex3f((tetMesh.vertexes[surf[j].y].x),
                   (tetMesh.vertexes[surf[j].y].y),
                   (tetMesh.vertexes[surf[j].y].z));
        glVertex3f((tetMesh.vertexes[surf[j].z].x),
                   (tetMesh.vertexes[surf[j].z].y),
                   (tetMesh.vertexes[surf[j].z].z));
    }
    glEnd();

    glColor3f(0.9f, 0.9f, 0.9f);
    //glDisable(GL_DEPTH_TEST);
    glLineWidth(0.1f);
    glBegin(GL_LINES);

    for(int j = 0; j < tetMesh.surfEdges.size(); j++)
    {
        //if ((tetMesh.surfEdges[j].x == 870 && tetMesh.surfEdges[j].y == 965) || (tetMesh.surfEdges[j].x == 965 && tetMesh.surfEdges[j].y == 870)) {
        //    glColor3f(0.9f, 0.1f, 0.1f);
        //    glLineWidth(3.4f);
        //}
        //else if ((tetMesh.surfEdges[j].x == 870 && tetMesh.surfEdges[j].y == 905) || (tetMesh.surfEdges[j].x == 905 && tetMesh.surfEdges[j].y == 870)) {
        //    glColor3f(0.9f, 0.9f, 0.1f);
        //    glLineWidth(3.4f);
        //}

        glVertex3f((tetMesh.vertexes[tetMesh.surfEdges[j].x].x),
                   (tetMesh.vertexes[tetMesh.surfEdges[j].x].y),
                   (tetMesh.vertexes[tetMesh.surfEdges[j].x].z));
        glVertex3f((tetMesh.vertexes[tetMesh.surfEdges[j].y].x),
                   (tetMesh.vertexes[tetMesh.surfEdges[j].y].y),
                   (tetMesh.vertexes[tetMesh.surfEdges[j].y].z));

        glColor3f(0.9f, 0.9f, 0.9f);
        glLineWidth(0.1f);
    }
    glEnd();

    //glColor3f(0.99f, 0.1f, 0.1f);
    ////glDisable(GL_DEPTH_TEST);
    //glPointSize(8);
    //glBegin(GL_POINTS);
    //glVertex3f((tetMesh.vertexes[2189].x), (tetMesh.vertexes[2189].y), (tetMesh.vertexes[2189].z));
    //glColor3f(0.99f, 0.99f, 0.1f);
    //glVertex3f((tetMesh.vertexes[870].x), (tetMesh.vertexes[870].y), (tetMesh.vertexes[870].z));
    //glVertex3f((tetMesh.vertexes[905].x), (tetMesh.vertexes[905].y), (tetMesh.vertexes[905].z));
    //glVertex3f((tetMesh.vertexes[965].x), (tetMesh.vertexes[965].y), (tetMesh.vertexes[965].z));
    //glEnd();
}

void draw_bvh()
{
    int num = (bvs.size() + 1) / 2;
    for(int j = 0; j < bvs.size(); j++)
    {
        int   i = j;
        float ox, oy, oz, bwidth, bheight, blength;
        ox      = (bvs[i].lower.x);
        oy      = (bvs[i].lower.y);
        oz      = (bvs[i].lower.z);
        bwidth  = (bvs[i].upper.x - bvs[i].lower.x);
        bheight = (bvs[i].upper.y - bvs[i].lower.y);
        blength = (bvs[i].upper.z - bvs[i].lower.z);
        draw_box3D(ox, oy, oz, bwidth, bheight, blength);
    }
}

int            counttt = 0;
std::vector<float3> getRenderGeometry(int& number)
{

    std::vector<double3> meshNormal(tetMesh.vertexNum, make_double3(0, 0, 0));
    number = tetMesh.surface.size();  //meshTemp.surfaceRender.size();
    std::vector<float3> pos_normal_color(3 * number * 3);

    for(int i = 0; i < number; i++)
    {
        //int tetId = meshTemp.surfaceRender[i][3];
        int v0 = tetMesh.surface[i].x;
        int v1 = tetMesh.surface[i].y;
        int v2 = tetMesh.surface[i].z;
        double3 vt0 = tetMesh.vertexes[v0];  // Vector3d(meshTemp.vertexes[v0][0], meshTemp.vertexes[v0][1], meshTemp.vertexes[v0][2]);
        double3 vt1 = tetMesh.vertexes[v1];  // Vector3d(meshTemp.vertexes[v1][0], meshTemp.vertexes[v1][1], meshTemp.vertexes[v1][2]);
        double3 vt2 = tetMesh.vertexes[v2];  // Vector3d(meshTemp.vertexes[v2][0], meshTemp.vertexes[v2][1], meshTemp.vertexes[v2][2]);
        double3 vec1 = __GEIGEN__::__minus(vt1, vt0);  //vt1 - vt0;
        double3 vec2 = __GEIGEN__::__minus(vt2, vt0);
        double3 normal =
            __GEIGEN__::__normalized(__GEIGEN__::__v_vec_cross(vec1, vec2));  //vec1.cross(vec2).normalized();

        pos_normal_color[i * 9]     = make_float3(vt0.x, vt0.y, vt0.z);
        pos_normal_color[i * 9 + 3] = make_float3(vt1.x, vt1.y, vt1.z);
        pos_normal_color[i * 9 + 6] = make_float3(vt2.x, vt2.y, vt2.z);

        pos_normal_color[i * 9 + 2] = make_float3(0.6875f, 0.51953f, 0.38671f);
        pos_normal_color[i * 9 + 5] = make_float3(0.6875f, 0.51953f, 0.38671f);
        pos_normal_color[i * 9 + 8] = make_float3(0.6875f, 0.51953f, 0.38671f);
        //}


        meshNormal[v0] = __GEIGEN__::__add(meshNormal[v0], normal);  //normal;
        meshNormal[v1] = __GEIGEN__::__add(meshNormal[v1], normal);
        meshNormal[v2] = __GEIGEN__::__add(meshNormal[v2], normal);
    }
    for(int i = 0; i < number; i++)

    {
        int v0 = tetMesh.surface[i].x;
        int v1 = tetMesh.surface[i].y;
        int v2 = tetMesh.surface[i].z;
        //meshNormal[v0].normalize(); meshNormal[v1].normalize(); meshNormal[v2].normalize();
        pos_normal_color[i * 9 + 1] =
            make_float3(meshNormal[v0].x, meshNormal[v0].y, meshNormal[v0].z);
        pos_normal_color[i * 9 + 4] =
            make_float3(meshNormal[v1].x, meshNormal[v1].y, meshNormal[v1].z);
        pos_normal_color[i * 9 + 7] =
            make_float3(meshNormal[v2].x, meshNormal[v2].y, meshNormal[v2].z);
    }

    return pos_normal_color;
}


void draw_Scene3D()
{
    //face.mesh3Ds[0] = mesh3d;
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    glClearColor(0.5f, 0.5f, 0.5f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glTranslatef(xTrans, yTrans, zTrans);
    glRotatef(xRot, 1.0f, 0.0f, 0.0f);
    glRotatef(yRot, 0.0f, 1.0f, 0.0f);

    //draw_box3D(-2, -1, -2, 4, 4, 4, 1);
    if(drawSurface)
    {
        draw_mesh3D();
    }
    if(drawbvh)
    {
        draw_bvh();
    }

    glPopMatrix();


    glutSwapBuffers();
    //glFlush();
}
double mfsum                   = 0;
double total_time              = 0;
int    total_cg_iterations     = 0;
int    total_newton_iterations = 0;
int    start                   = -1;

void saveScreenPic(const std::string& path)
{
    std::stringstream ss;
    ss << path;
    ss.fill('0');
    ss.width(5);
    ss << step;
    std::string file_path = ss.str();

    SaveScreenShot(window_width, window_height, file_path);
}

void initFEM(tetrahedra_obj& mesh)
{

    double massSum   = 0;
    double volumeSum = 0;
    //float  angleX = FEM::PI / 4, angleY = -FEM::PI / 4, angleZ = FEM::PI / 2;
    //__GEIGEN__::Matrix3x3d rotation, rotationZ, rotationY, rotationX, eigenTest;
    //__GEIGEN__::__set_Mat_val(rotation, 1, 0, 0, 0, 1, 0, 0, 0, 1);
    //__GEIGEN__::__set_Mat_val(
    //    rotationZ, cos(angleZ), -sin(angleZ), 0, sin(angleZ), cos(angleZ), 0, 0, 0, 1);
    //__GEIGEN__::__set_Mat_val(
    //    rotationY, cos(angleY), 0, -sin(angleY), 0, 1, 0, sin(angleY), 0, cos(angleY));
    //__GEIGEN__::__set_Mat_val(
    //    rotationX, 1, 0, 0, 0, cos(angleX), -sin(angleX), 0, sin(angleX), cos(angleX));


    ipc.lengthRateLame = ipc.YoungModulus / (2 * (1 + ipc.PoissonRate));
    ipc.volumeRateLame = ipc.YoungModulus * ipc.PoissonRate
                         / ((1 + ipc.PoissonRate) * (1 - 2 * ipc.PoissonRate));
    ipc.lengthRate   = 4 * ipc.lengthRateLame / 3;
    ipc.volumeRate   = ipc.volumeRateLame + 5 * ipc.lengthRateLame / 6;
    ipc.stretchStiff = ipc.clothYoungModulus / (2 * (1 + ipc.PoissonRate));

    ipc.bendStiff = ipc.bendYoungModulus * pow(ipc.clothThickness, 3)
                    / (24 * (1 - ipc.PoissonRate * ipc.PoissonRate));

    ipc.shearStiff = 0.03 * ipc.stretchStiff * ipc.strainRate;

    printf("ipc.shearStiff: %f\n", ipc.shearStiff);


    for(int i = 0; i < mesh.tetrahedraNum; i++)
    {
        __GEIGEN__::Matrix3x3d DM;
        __calculateDms3D_double(mesh.vertexes.data(), mesh.tetrahedras[i], DM);  //calculateDms3D_double(mesh.vertexes, mesh.tetrahedras[i], 0);

        __GEIGEN__::Matrix3x3d DM_inverse;
        __GEIGEN__::__Inverse(DM, DM_inverse);

        double vlm = calculateVolum(mesh.vertexes.data(), mesh.tetrahedras[i]);

        mesh.masses[mesh.tetrahedras[i].x] += vlm * ipc.density / 4;
        mesh.masses[mesh.tetrahedras[i].y] += vlm * ipc.density / 4;
        mesh.masses[mesh.tetrahedras[i].z] += vlm * ipc.density / 4;
        mesh.masses[mesh.tetrahedras[i].w] += vlm * ipc.density / 4;

        massSum += vlm * ipc.density;
        volumeSum += vlm;
        mesh.DM_inverse.push_back(DM_inverse);
        mesh.volum.push_back(vlm);


        double lengthRateLame =
            mesh.vert_youngth_modules[i] / (2 * (1 + ipc.PoissonRate));
        double volumeRateLame = mesh.vert_youngth_modules[i] * ipc.PoissonRate
                                / ((1 + ipc.PoissonRate) * (1 - 2 * ipc.PoissonRate));
        double lengthRate = 4 * lengthRateLame / 3;
        double volumeRate = volumeRateLame + 5 * lengthRateLame / 6;

        mesh.lengthRate.push_back(lengthRate);
        mesh.volumeRate.push_back(volumeRate);
    }

    for(int i = 0; i < mesh.triangles.size(); i++)
    {
        __GEIGEN__::Matrix2x2d DM;
        __calculateDm2D_double(mesh.vertexes.data(), mesh.triangles[i], DM);

        __GEIGEN__::Matrix2x2d DM_inverse;
        __GEIGEN__::__Inverse2x2(DM, DM_inverse);

        double area = calculateArea(mesh.vertexes.data(), mesh.triangles[i]);
        area *= ipc.clothThickness;
        mesh.area.push_back(area);


        mesh.masses[mesh.triangles[i].x] += ipc.clothDensity * area / 3;
        mesh.masses[mesh.triangles[i].y] += ipc.clothDensity * area / 3;
        mesh.masses[mesh.triangles[i].z] += ipc.clothDensity * area / 3;

        massSum += area * ipc.clothDensity;
        volumeSum += area;
        mesh.tri_DM_inverse.push_back(DM_inverse);
    }

    mesh.meanMass = massSum / mesh.vertexNum;
    printf("meanMass: %f\n", mesh.meanMass);
    mesh.meanVolum = volumeSum / mesh.vertexNum;
}

void DefaultSettings()
{
    // global settings
    ipc.density        = 1e3;
    ipc.PoissonRate    = 0.49;
    //ipc.lengthRateLame = ipc.YoungModulus / (2 * (1 + ipc.PoissonRate));
    //ipc.volumeRateLame = ipc.YoungModulus * ipc.PoissonRate
    //                     / ((1 + ipc.PoissonRate) * (1 - 2 * ipc.PoissonRate));
    //ipc.lengthRate        = 4 * ipc.lengthRateLame / 3;
    //ipc.volumeRate        = ipc.volumeRateLame + 5 * ipc.lengthRateLame / 6;
    ipc.frictionRate      = 0.4;
    ipc.gd_frictionRate   = 0.4;
    ipc.clothThickness    = 1e-3;
    ipc.clothYoungModulus = 1e6;
    ipc.bendYoungModulus  = 1e5;
    //ipc.stretchStiff      = ipc.clothYoungModulus / (2 * (1 + ipc.PoissonRate));
    //ipc.shearStiff        = ipc.stretchStiff * 0.3;
    ipc.clothDensity      = 2e2;
    ipc.strainRate        = 100;
    ipc.softMotionRate    = 1e0;
    ipc.bendStiff         = 3e-4;
    ipc.Newton_solver_threshold = 1e-2;
    ipc.pcg_threshold           = 1e-4;
    ipc.IPC_dt                  = 1e-2;
    ipc.relative_dhat           = 1e-3;
    //ipc.bendStiff = ipc.bendYoungModulus * pow(ipc.clothThickness, 3)
    //                / (24 * (1 - ipc.PoissonRate * ipc.PoissonRate));
    //ipc.shearStiff = 0.03 * ipc.stretchStiff * ipc.strainRate;
}
//int  meshids = 0;
void LoadSettings()
{
    bool successfulRead = false;

    //read file
    std::ifstream infile;


    std::string DEFAULT_CONFIG_FILE = assets_dir + "scene/parameterSetting.txt";


    infile.open(DEFAULT_CONFIG_FILE, std::ifstream::in);
    if(successfulRead = infile.is_open())
    {
        int  tempEnum;
        char ignoreToken[256];

        // global settings:
        infile >> ignoreToken >> ipc.density;
        infile >> ignoreToken >> ipc.PoissonRate;
        infile >> ignoreToken >> ipc.frictionRate;
        infile >> ignoreToken >> ipc.gd_frictionRate;
        infile >> ignoreToken >> ipc.clothThickness;
        infile >> ignoreToken >> ipc.clothYoungModulus;
        infile >> ignoreToken >> ipc.bendYoungModulus;
        //infile >> ignoreToken >> ipc.shearStiff;
        infile >> ignoreToken >> ipc.clothDensity;
        infile >> ignoreToken >> ipc.strainRate;
        infile >> ignoreToken >> ipc.softMotionRate;
        //infile >> ignoreToken >> ipc.bendStiff;
        infile >> ignoreToken >> collision_detection_buff_scale;
        infile >> ignoreToken >> motion_rate;
        infile >> ignoreToken >> ipc.IPC_dt;
        infile >> ignoreToken >> ipc.pcg_threshold;
        infile >> ignoreToken >> ipc.Newton_solver_threshold;
        infile >> ignoreToken >> ipc.relative_dhat;
        //infile >> ignoreToken >> meshids;



        //ipc.shearStiff =
        infile.close();
    }

    if(!successfulRead)
    {
        std::cerr << "Waning: failed loading settings, set to defaults." << std::endl;
        DefaultSettings();
    }
}

void set_case1()
{
    double                    dist       = 0.2;
    int                       count      = 4;
    int                       count_Y    = 4;
    double                    fem_height = -0.8;
    double                    abd_height = -0.6;
    gipc::SimpleSceneImporter importer;

    linear_system_buff_scale = 1.0;

    double Youngth_Modulus = 1e4;
    for(int k = 0; k < count_Y; ++k)
    {
        for(int i = 0; i < count; i++)
        {
            for(int j = 0; j < count; j++)
            {

                gipc::Vector2 ij{i, j};
                gipc::Vector2 pos =
                    ij * dist - gipc::Vector2::Ones() * dist * (count - 1) / 2.0;

                double3 position_offset =
                    make_double3(-pos.x(), -abd_height - 2 * dist * k, -pos.y());
                double          scale     = 0.4;
                Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
                transform.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * scale;
                transform.block<3, 1>(0, 3) = -Eigen::Vector3d(
                    position_offset.x, position_offset.y, position_offset.z);

                importer.load_geometry(tetMesh,
                                       3,
                                       gipc::BodyType::ABD,
                                       transform,
                                       1e5,
                                       assets_dir + "tetMesh/cube.msh",
                                       ipc.pcg_data.P_type);
            }
        }
    }

    for(int k = 0; k < count_Y; ++k)
    {
        for(int i = 0; i < count; i++)
        {
            for(int j = 0; j < count; j++)
            {

                gipc::Vector2 ij{i, j};
                gipc::Vector2 pos =
                    ij * dist - gipc::Vector2::Ones() * dist * (count - 1) / 2.0;

                double3 position_offset =
                    make_double3(-pos.x(), -fem_height - 2 * dist * k, -pos.y());
                double          scale     = 0.4;
                Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
                transform.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * scale;
                transform.block<3, 1>(0, 3) = -Eigen::Vector3d(
                    position_offset.x, position_offset.y, position_offset.z);

                importer.load_geometry(tetMesh,
                                       3,
                                       gipc::BodyType::FEM,
                                       transform,
                                       Youngth_Modulus,
                                       assets_dir + "tetMesh/cube.msh",
                                       ipc.pcg_data.P_type);
            }
        }
    }
}


void set_case2()
{
    gipc::SimpleSceneImporter importer;
    collision_detection_buff_scale = runtime_options.figure12_collision_buffer_scale;
    double                    scale           = 0.2;
    double3                   position_offset = make_double3(0, -0.5, 0);
    Eigen::Matrix4d           transform       = Eigen::Matrix4d::Identity();
    transform.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * scale;
    transform.block<3, 1>(0, 3) =
        -Eigen::Vector3d(position_offset.x, position_offset.y, position_offset.z);

    linear_system_buff_scale = runtime_options.figure12_linear_system_buffer_scale;
    std::string mesh0_path      = assets_dir + "tetMesh/bunny2.msh";
    const auto first_bunny_body_type = runtime_options.body_mode == "hybrid-abd"
                                           ? gipc::BodyType::ABD
                                           : gipc::BodyType::FEM;
    importer.load_geometry(tetMesh,
                           3,
                           first_bunny_body_type,
                           transform,
                           figure12_primary_bunny_young_modulus,
                           mesh0_path,
                           ipc.pcg_data.P_type);

    if(runtime_options.figure12_bunny_count == 2)
    {
        position_offset             = make_double3(0, 0.65, 0);
        transform                   = Eigen::Matrix4d::Identity();
        transform.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * scale;
        transform.block<3, 1>(0, 3) =
            -Eigen::Vector3d(position_offset.x, position_offset.y, position_offset.z);

        std::string mesh1_path = mesh0_path;
        importer.load_geometry(tetMesh,
                               3,
                               gipc::BodyType::FEM,
                               transform,
                               figure12_secondary_bunny_young_modulus,
                               mesh1_path,
                               ipc.pcg_data.P_type);
    }


    position_offset             = make_double3(0, 0, 0);
    transform                   = Eigen::Matrix4d::Identity();
    transform.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * 1.0;
    transform.block<3, 1>(0, 3) =
        -Eigen::Vector3d(position_offset.x, position_offset.y, position_offset.z);
    std::string mesh2_path = runtime_options.cloth_mesh.empty()
                                 ? assets_dir + "triMesh/cloth_high.obj"
                                 : runtime_options.cloth_mesh;

    const auto cloth_vertex_offset   = tetMesh.vertexes.size();
    const auto cloth_triangle_offset = tetMesh.triangles.size();
    importer.load_geometry(tetMesh,
                           2,
                           gipc::BodyType::FEM,
                           transform,
                           runtime_options.young_modulus,
                           mesh2_path,
                           ipc.pcg_data.P_type);
    figure12_cloth_vertex_count =
        static_cast<int>(tetMesh.vertexes.size() - cloth_vertex_offset);
    figure12_cloth_triangle_count =
        static_cast<int>(tetMesh.triangles.size() - cloth_triangle_offset);
}

void set_case3()
{

    gipc::SimpleSceneImporter importer{assets_dir + "scene/json/wrecking-ball-simple.json",
                                       assets_dir + "tetMesh/wrecking-ball-mesh/",
                                       gipc::BodyType::ABD};
    linear_system_buff_scale = 1.0;
    importer.import_scene(tetMesh);
}

void set_case4()
{
    ipc.pcg_data.P_type = 0;
    linear_system_buff_scale = 1.0;
    gipc::SimpleSceneImporter importer;
    double                    scale           = 0.6;
    double3                   position_offset = make_double3(0, 1.0, 0);

    using Transform = Eigen::Transform<double, 3, Eigen::Affine>;
    Transform t     = Transform::Identity();
    t.translate(Eigen::Vector3d{0, 1.0, 0});
    t.scale(scale);
    t.rotate(Eigen::AngleAxisd(3.1415926 / 2, Eigen::Vector3d::UnitX()));
    Eigen::Matrix4d transform = t.matrix();

    std::string mesh_path = assets_dir + "triMesh/cloth_high.obj";
    importer.load_geometry(tetMesh,
                           2,
                           gipc::BodyType::FEM,
                           transform,
                           1e4,
                           mesh_path,
                           ipc.pcg_data.P_type);

    int          fixed_vertex_num = 0;
    const double eps              = 1e-4;
    double       max_y            = tetMesh.maxTConer.y;
    double       min_x            = tetMesh.minTConer.x;
    double       max_x            = tetMesh.maxTConer.x;
    for(int i = 0; i < tetMesh.vertexNum; i++)
    {
        if(tetMesh.vertexes[i].y > max_y - eps
           && (tetMesh.vertexes[i].x < min_x + eps || tetMesh.vertexes[i].x > max_x - eps))
        {
            tetMesh.boundaryTypies[i] = 1;
            fixed_vertex_num++;
        }
    }
    std::cout << "fixed vertex num: " << fixed_vertex_num << std::endl;
}

void set_case5()
{
    ipc.pcg_data.P_type = 1;
    linear_system_buff_scale = 2.0;
    gipc::SimpleSceneImporter importer;
    double                    scale = 1.0;
    Eigen::Vector3d           position_offset{0, 0, 0};

    using Transform = Eigen::Transform<double, 3, Eigen::Affine>;
    Transform t     = Transform::Identity();
    t.translate(position_offset);
    t.scale(scale);
    Eigen::Matrix4d transform = t.matrix();

    std::string mesh_path       = assets_dir + "tetMesh/high_mat.msh";
    double Youngth_Modulus = 1e4;
    ipc.PoissonRate        = 0.48;
    importer.load_geometry(tetMesh,
                           3,
                           gipc::BodyType::FEM,
                           transform,
                           Youngth_Modulus,
                           mesh_path,
                           ipc.pcg_data.P_type);

    // no gravity
    for(int i = 0; i < tetMesh.vertexes.size(); i++)
    {
        tetMesh.apply_gravity[i] = 0;
    }

    const double eps = 1e-4;
    for(int i = 0; i < tetMesh.vertexNum; i++)
    {
        if(tetMesh.vertexes[i].x < -0.5 + eps || tetMesh.vertexes[i].x > 0.5 - eps)
        {
            tetMesh.targetIndex.push_back(i);
            tetMesh.targetPos.push_back(tetMesh.vertexes[i]);
        }
    }
    tetMesh.softNum = tetMesh.targetIndex.size();
    std::cout << "soft constraint num: " << tetMesh.softNum << std::endl;
    ipc.softMotionRate = 1;

    const double angular_vel = 3.14159265358979323846/5;
    d_tetMesh.update_soft_constraint_functor =
        [angular_vel](double3 vertex, int step_id, double ipc_dt) -> double3
    {
        double3 rotated_vertex = vertex;
        if(vertex.x < 0)
        {
            // rotate along x axis clockwise
            rotated_vertex = {vertex.x,
                              vertex.y * std::cos(angular_vel * ipc_dt)
                                  - vertex.z * std::sin(angular_vel * ipc_dt),
                              vertex.y * std::sin(angular_vel * ipc_dt)
                                  + vertex.z * std::cos(angular_vel * ipc_dt)};
        }
        if(vertex.x > 0)
        {
            // rotate along x axis counterclockwise
            rotated_vertex = {vertex.x,
                              vertex.y * std::cos(-angular_vel * ipc_dt)
                                  - vertex.z * std::sin(-angular_vel * ipc_dt),
                              vertex.y * std::sin(-angular_vel * ipc_dt)
                                  + vertex.z * std::cos(-angular_vel * ipc_dt)};
        }
        return rotated_vertex;
    };
}

void set_case_stiff_bunny_drop()
{
    std::cerr << "[headless] bunny_config" << std::endl;
    ipc.pcg_data.P_type       = 1;
    linear_system_buff_scale  = 1.5;
    collision_detection_buff_scale = 2.0;

    gipc::SimpleSceneImporter importer;
    using Transform = Eigen::Transform<double, 3, Eigen::Affine>;
    Transform transform = Transform::Identity();
    transform.translate(Eigen::Vector3d(0.0, -0.70, 0.0));
    transform.scale(0.20);

    std::cerr << "[headless] bunny_load " << runtime_options.tet_mesh << std::endl;
    importer.load_geometry(tetMesh,
                           3,
                           gipc::BodyType::FEM,
                           transform.matrix(),
                           runtime_options.young_modulus,
                           runtime_options.tet_mesh,
                           ipc.pcg_data.P_type);
    std::cerr << "[headless] bunny_loaded vertices=" << tetMesh.vertexNum << std::endl;
    ipc.relative_dhat = 1e-3;
}

void set_case6()
{
    linear_system_buff_scale = 2.0;
    ipc.pcg_data.P_type = 1;
    double scale      = 0.3;
    double dist       = scale / 2;
    int    count      = 8;
    int    count_Y    = 15;
    double fem_height = global_offset + 1 - 0.8;
    double abd_height = fem_height - dist;


    for(int k = 0; k < count_Y; ++k)
    {
        for(int i = 0; i < count; i++)
        {
            for(int j = 0; j < count; j++)
            {

                gipc::Vector2 ij{i, j};
                gipc::Vector2 pos =
                    ij * dist - gipc::Vector2::Ones() * dist * (count - 1) / 2.0;


                double3 position_offset =
                    double3{-pos.x(), -abd_height - 2 * dist * k, -pos.y()};
                Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
                transform.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * scale;
                transform.block<3, 1>(0, 3) = -Eigen::Vector3d(
                    position_offset.x, position_offset.y, position_offset.z);
                tetMesh.load_tetrahedraMesh(assets_dir + "tetMesh/cube.msh",
                                            transform,
                                            1e6,
                                            gipc::BodyType::ABD);
            }
        }
    }

    for(int k = 0; k < count_Y; ++k)
    {
        for(int i = 0; i < count; i++)
        {
            for(int j = 0; j < count; j++)
            {
                gipc::Vector2 ij{i, j};
                gipc::Vector2 pos =
                    ij * dist - gipc::Vector2::Ones() * dist * (count - 1) / 2.0;

                double3 position_offset =
                    double3{-pos.x(), -fem_height - 2 * dist * k, -pos.y()};
                Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
                transform.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * scale;
                transform.block<3, 1>(0, 3) = -Eigen::Vector3d(
                    position_offset.x, position_offset.y, position_offset.z);
                tetMesh.load_tetrahedraMesh(assets_dir + "tetMesh/cube.msh",
                                            transform,
                                            5e4,
                                            gipc::BodyType::ABD);
            }
        }
    }
    
    gipc::SimpleSceneImporter importer;
    using Transform = Eigen::Transform<double, 3, Eigen::Affine>;
    Transform t     = Transform::Identity();
    t.scale(1.5);
    t.translate(Eigen::Vector3d(0, 0.35, 0));
    std::string mesh_path = assets_dir + "triMesh/cloth_high.obj";
    importer.load_geometry(tetMesh,
                           2,
                           gipc::BodyType::FEM,
                           t.matrix(),
                           1e4,
                           mesh_path,
                           ipc.pcg_data.P_type);

    const double eps = 1e-4;
    for(int i = 0; i < tetMesh.vertexNum; i++)
    {
        if(tetMesh.vertexes[i].x < -1.5 + eps || tetMesh.vertexes[i].x > 1.5 - eps)
        {
            tetMesh.boundaryTypies[i] = 1;
        }
    }

    ipc.relative_dhat = 1e-3;
    ipc.strainRate    = 1e6;
}

void setMAS_partition()
{
    tetMesh.partId_map_real.resize(tetMesh.part_offset * BANKSIZE, -1);
    tetMesh.real_map_partId.resize(tetMesh.partId.size());
    int index = 0;
    for(int i = 0; i < tetMesh.partId.size(); i++)
    {
        tetMesh.partId_map_real[BANKSIZE * tetMesh.partId[i] + index] = i;
        index++;
        if(i <= tetMesh.partId.size() - 2)
        {
            if(tetMesh.partId[i + 1] != tetMesh.partId[i])
            {
                index = 0;
            }
        }
    }
    index = 0;
    for(int i = 0; i < tetMesh.partId_map_real.size(); i++)
    {

        if(tetMesh.partId_map_real[i] == index)
        {
            tetMesh.real_map_partId[index] = i;
            index++;
        }
    }
}

void set_case_fig15_cloth_abd_scaled()
{
    gipc::SimpleSceneImporter importer;
    collision_detection_buff_scale = 1.0;
    linear_system_buff_scale = 1.0;

    Eigen::Matrix4d sphere_transform = Eigen::Matrix4d::Identity();
    sphere_transform.block<3,3>(0,0) = Eigen::Matrix3d::Identity()*0.2;
    sphere_transform(1,3) = -0.15;
    importer.load_geometry(
        tetMesh,3,gipc::BodyType::ABD,sphere_transform,1e7,
        assets_dir+"tetMesh/wrecking-ball-mesh/wrecking-ball/ball.obj_.msh",
        ipc.pcg_data.P_type,BodyBoundaryType::Fixed);

    Eigen::Matrix4d cloth_transform = Eigen::Matrix4d::Identity();
    cloth_transform.block<3,3>(0,0) = Eigen::Matrix3d::Identity()*0.6;
    cloth_transform(1,3) = 0.35;
    const std::string cloth_path = runtime_options.cloth_mesh.empty()
                                       ? assets_dir+"triMesh/cloth_high.obj"
                                       : runtime_options.cloth_mesh;
    const auto vertex_offset=tetMesh.vertexes.size();
    const auto triangle_offset=tetMesh.triangles.size();
    importer.load_geometry(tetMesh,2,gipc::BodyType::FEM,cloth_transform,
                           runtime_options.young_modulus,cloth_path,ipc.pcg_data.P_type);
    figure15_cloth_vertex_count=static_cast<int>(tetMesh.vertexes.size()-vertex_offset);
    figure15_cloth_triangle_count=static_cast<int>(tetMesh.triangles.size()-triangle_offset);
    ipc.clothYoungModulus=runtime_options.young_modulus;
}

namespace
{
uint32_t expand_morton_10(uint32_t value)
{
    value &= 0x000003ffu;
    value = (value | (value << 16)) & 0x030000FFu;
    value = (value | (value << 8)) & 0x0300F00Fu;
    value = (value | (value << 4)) & 0x030C30C3u;
    value = (value | (value << 2)) & 0x09249249u;
    return value;
}

uint32_t morton_code(const double3& p, const double3& lower, const double3& upper)
{
    auto quantize = [](double value, double lo, double hi) {
        const double extent = hi - lo;
        const double t = extent > 0.0 ? (value - lo) / extent : 0.0;
        return static_cast<uint32_t>(std::clamp(t, 0.0, 1.0) * 1023.0 + 0.5);
    };
    const uint32_t x = expand_morton_10(quantize(p.x, lower.x, upper.x));
    const uint32_t y = expand_morton_10(quantize(p.y, lower.y, upper.y));
    const uint32_t z = expand_morton_10(quantize(p.z, lower.z, upper.z));
    return x | (y << 1) | (z << 2);
}
}  // namespace

void set_gpu_mas32_partition()
{
    const int fem_offset = tetMesh.abd_vertexOffset;
    const int fem_count  = static_cast<int>(tetMesh.vertexes.size()) - fem_offset;
    if(fem_count <= 0)
        return;

    double3 lower = tetMesh.vertexes[fem_offset];
    double3 upper = lower;
    for(int local = 1; local < fem_count; ++local)
    {
        const auto& p = tetMesh.vertexes[fem_offset + local];
        lower.x = std::min(lower.x, p.x);
        lower.y = std::min(lower.y, p.y);
        lower.z = std::min(lower.z, p.z);
        upper.x = std::max(upper.x, p.x);
        upper.y = std::max(upper.y, p.y);
        upper.z = std::max(upper.z, p.z);
    }

    std::vector<int> order(fem_count);
    for(int i = 0; i < fem_count; ++i)
        order[i] = i;
    std::stable_sort(order.begin(), order.end(), [&](int lhs, int rhs) {
        const uint32_t lhs_code = morton_code(tetMesh.vertexes[fem_offset + lhs], lower, upper);
        const uint32_t rhs_code = morton_code(tetMesh.vertexes[fem_offset + rhs], lower, upper);
        return lhs_code == rhs_code ? lhs < rhs : lhs_code < rhs_code;
    });

    tetMesh.part_offset = (fem_count + GPU_MAS_BANKSIZE - 1) / GPU_MAS_BANKSIZE;
    tetMesh.partId_map_real.assign(tetMesh.part_offset * GPU_MAS_BANKSIZE, -1);
    tetMesh.real_map_partId.resize(fem_count);
    for(int slot = 0; slot < fem_count; ++slot)
    {
        const int real = order[slot];
        tetMesh.partId_map_real[slot] = real;
        tetMesh.real_map_partId[real] = slot;
    }
    std::cout << "traditional GPU MAS32 Morton groups: " << tetMesh.part_offset
              << ", FEM vertices: " << fem_count << std::endl;
}

void initScene()
{
    std::cerr << "[headless] ensure_metis_dir " << metis_dir << std::endl;
    std::filesystem::exists(metis_dir) || std::filesystem::create_directory(metis_dir);
    ipc.pcg_data.P_type = runtime_options.preconditioner == "cemas16"
                              ? 1
                              : (runtime_options.preconditioner == "gpu-mas" ? 2 : 0);

    if(runtime_options.scene == "stiff-bunny-drop")
    {
        set_case_stiff_bunny_drop();
    }
    else if(runtime_options.scene == "paper-fig12-coupling-scaled")
    {
        set_case2();
    }
    else if(runtime_options.scene == "paper-fig15-cloth-abd-scaled")
    {
        set_case_fig15_cloth_abd_scaled();
    }
    else
    {
    int scene_no = 1;
    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //!!!!!!!!!!!!!!!!ABD must be loaded before FEM!!!!!!!!!!!!!!!!!!
    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    switch(scene_no)
    {
        case 0:  // box pipe
            set_case1();
            break;
        case 1:  // soft-rigid-cloth coupling
            set_case2();
            break;
        case 2:  //wrecking ball case
            set_case3();
            break;
        case 3:  //fixed cloth
            set_case4();
            break;
        case 4:  //twisting mat
            set_case5();
            break;
        case 5:  //box pipe large scale and cloth
            set_case6();
            break;
    }
    }


    std::cerr << "[headless] set_mas_partition" << std::endl;
    if(ipc.pcg_data.P_type == 1)
        setMAS_partition();
    else if(ipc.pcg_data.P_type == 2)
        set_gpu_mas32_partition();


    std::cerr << "[headless] get_surface" << std::endl;
    tetMesh.getSurface();

    std::cerr << "[headless] init_fem" << std::endl;
    initFEM(tetMesh);
    //device_TetraData d_tetMesh;
    d_tetMesh.Malloc_DEVICE_MEM(tetMesh.vertexNum,
                                tetMesh.tetrahedraNum,
                                tetMesh.triangleNum,
                                tetMesh.softNum,
                                tetMesh.tri_edges.size(),
                                tetMesh.abd_fem_count_info.total_body_num());

    CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.masses,
                              tetMesh.masses.data(),
                              tetMesh.vertexNum * sizeof(double),
                              cudaMemcpyHostToDevice));

    CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.apply_gravity,
                              tetMesh.apply_gravity.data(),
                              tetMesh.vertexNum * sizeof(int),
                              cudaMemcpyHostToDevice));

    CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.lengthRate,
                              tetMesh.lengthRate.data(),
                              tetMesh.tetrahedraNum * sizeof(double),
                              cudaMemcpyHostToDevice));

    CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.volumeRate,
                              tetMesh.volumeRate.data(),
                              tetMesh.tetrahedraNum * sizeof(double),
                              cudaMemcpyHostToDevice));


    CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.volum,
                              tetMesh.volum.data(),
                              tetMesh.tetrahedraNum * sizeof(double),
                              cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.vertexes,
                              tetMesh.vertexes.data(),
                              tetMesh.vertexNum * sizeof(double3),
                              cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.o_vertexes,
                              tetMesh.vertexes.data(),
                              tetMesh.vertexNum * sizeof(double3),
                              cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.tetrahedras,
                              tetMesh.tetrahedras.data(),
                              tetMesh.tetrahedraNum * sizeof(uint4),
                              cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.DmInverses,
                              tetMesh.DM_inverse.data(),
                              tetMesh.tetrahedraNum * sizeof(__GEIGEN__::Matrix3x3d),
                              cudaMemcpyHostToDevice));

    CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.BoundaryType,
                              tetMesh.boundaryTypies.data(),
                              tetMesh.vertexNum * sizeof(int),
                              cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.velocities,
                              tetMesh.velocities.data(),
                              tetMesh.vertexNum * sizeof(double3),
                              cudaMemcpyHostToDevice));


    CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.targetIndex,
                              tetMesh.targetIndex.data(),
                              tetMesh.softNum * sizeof(uint32_t),
                              cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.targetVert,
                              tetMesh.targetPos.data(),
                              tetMesh.softNum * sizeof(double3),
                              cudaMemcpyHostToDevice));

    d_tetMesh.host_target_indices  = tetMesh.targetIndex;
    d_tetMesh.host_target_vertices = tetMesh.targetPos;

    CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.triDmInverses,
                              tetMesh.tri_DM_inverse.data(),
                              tetMesh.triangleNum * sizeof(__GEIGEN__::Matrix2x2d),
                              cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.area,
                              tetMesh.area.data(),
                              tetMesh.triangleNum * sizeof(double),
                              cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.triangles,
                              tetMesh.triangles.data(),
                              tetMesh.triangleNum * sizeof(uint3),
                              cudaMemcpyHostToDevice));

    CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.tri_edges,
                              tetMesh.tri_edges.data(),
                              tetMesh.tri_edges.size() * sizeof(uint2),
                              cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.tri_edge_adj_vertex,
                              tetMesh.tri_edges_adj_points.data(),
                              tetMesh.tri_edges.size() * sizeof(uint2),
                              cudaMemcpyHostToDevice));

    CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.body_id_to_boundary_type,
                              tetMesh.body_id_to_is_fixed.data(),
                              tetMesh.body_id_to_is_fixed.size() * sizeof(int),
                              cudaMemcpyHostToDevice));

    CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.point_id_to_body_id,
                              tetMesh.point_id_to_body_id.data(),
                              tetMesh.point_id_to_body_id.size() * sizeof(int),
                              cudaMemcpyHostToDevice));

    CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.tet_id_to_body_id,
                              tetMesh.tet_id_to_body_id.data(),
                              tetMesh.tet_id_to_body_id.size() * sizeof(int),
                              cudaMemcpyHostToDevice));


    printf("stretchStiff:  %f,  shearStiff:   %f\n", ipc.stretchStiff, ipc.shearStiff);

    ipc.vertexNum      = tetMesh.vertexNum;
    ipc.tetrahedraNum  = tetMesh.tetrahedraNum;
    ipc._vertexes      = d_tetMesh.vertexes;
    ipc._rest_vertexes = d_tetMesh.rest_vertexes;
    ipc.surf_vertexNum = tetMesh.surfVerts.size();
    ipc.surface_Num    = tetMesh.surface.size();
    ipc.edge_Num       = tetMesh.surfEdges.size();
    ipc.tri_edge_num   = tetMesh.tri_edges.size();

    //ipc.IPC_dt = 0.01 / 1.0;//1.0 / 30;//1.0 / 100;
    ipc.MAX_CCD_COLLITION_PAIRS_NUM =
        1 * collision_detection_buff_scale
        * (((double)(ipc.surface_Num * 15 + ipc.edge_Num * 10))
           * std::max((ipc.IPC_dt / 0.01), 2.0));
    ipc.MAX_COLLITION_PAIRS_NUM = (ipc.surf_vertexNum * 3 + ipc.edge_Num * 2)
                                  * 3 * collision_detection_buff_scale;

    ipc.triangleNum        = tetMesh.triangleNum;
    ipc.targetVert         = d_tetMesh.targetVert;
    ipc.targetInd          = d_tetMesh.targetIndex;
    ipc.softNum            = tetMesh.softNum;
    ipc.abd_fem_count_info = tetMesh.abd_fem_count_info;

    std::cout << "ABD FEM count info: \n"
              << ipc.abd_fem_count_info << std::endl;


    printf("vertNum: %d      tetraNum: %d      faceNum: %d\n",
           ipc.vertexNum,
           ipc.tetrahedraNum,
           ipc.surface_Num);
    printf("surfVertNum: %d      surfEdgesNum: %d\n", ipc.surf_vertexNum, ipc.edge_Num);
    printf("maxCollisionPairsNum_CCD: %d      maxCollisionPairsNum: %d\n",
           ipc.MAX_CCD_COLLITION_PAIRS_NUM,
           ipc.MAX_COLLITION_PAIRS_NUM);

    //ipc.USE_MAS = false;
    ipc.MALLOC_DEVICE_MEM();

    CUDA_SAFE_CALL(cudaMemcpy(
        ipc._faces, tetMesh.surface.data(), ipc.surface_Num * sizeof(uint3), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(
        ipc._edges, tetMesh.surfEdges.data(), ipc.edge_Num * sizeof(uint2), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(ipc._surfVerts,
                              tetMesh.surfVerts.data(),
                              ipc.surf_vertexNum * sizeof(uint32_t),
                              cudaMemcpyHostToDevice));
    ipc.initBVH(d_tetMesh.BoundaryType, d_tetMesh.point_id_to_body_id);

    if(ipc.pcg_data.P_type == 1)
    {
        int neighborListSize = tetMesh.getVertNeighbors();
        ipc.pcg_data.MP.initPreconditioner_Neighbor(ipc.vertexNum - tetMesh.abd_vertexOffset,
                                                    tetMesh.abd_vertexOffset,
                                                    neighborListSize,
                                                    ipc._collisonPairs,
                                                    tetMesh.part_offset * BANKSIZE);

        ipc.pcg_data.MP.neighborListSize = neighborListSize;
        CUDA_SAFE_CALL(cudaMemcpy(ipc.pcg_data.MP.d_neighborListInit,
                                  tetMesh.neighborList.data(),
                                  neighborListSize * sizeof(unsigned int),
                                  cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(ipc.pcg_data.MP.d_neighborStart,
                                  tetMesh.neighborStart.data(),
                                  (ipc.vertexNum - tetMesh.abd_vertexOffset) * sizeof(unsigned int),
                                  cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(ipc.pcg_data.MP.d_neighborNumInit,
                                  tetMesh.neighborNum.data(),
                                  (ipc.vertexNum - tetMesh.abd_vertexOffset) * sizeof(unsigned int),
                                  cudaMemcpyHostToDevice));

        CUDA_SAFE_CALL(cudaMemcpy(ipc.pcg_data.MP.d_partId_map_real,
                                  tetMesh.partId_map_real.data(),
                                  tetMesh.part_offset * BANKSIZE * sizeof(int),
                                  cudaMemcpyHostToDevice));

        CUDA_SAFE_CALL(cudaMemcpy(ipc.pcg_data.MP.d_real_map_partId,
                                  tetMesh.real_map_partId.data(),
                                  tetMesh.real_map_partId.size() * sizeof(int),
                                  cudaMemcpyHostToDevice));

        ipc.pcg_data.MP.initPreconditioner_Matrix();
    }
    else if(ipc.pcg_data.P_type == 2)
    {
        auto& mas32 = ipc.pcg_data.traditional_mas32;
        int neighborListSize = tetMesh.getVertNeighbors();
        mas32.initPreconditioner_Neighbor(ipc.vertexNum - tetMesh.abd_vertexOffset,
                                          tetMesh.abd_vertexOffset,
                                          neighborListSize,
                                          ipc._collisonPairs,
                                          tetMesh.part_offset * GPU_MAS_BANKSIZE);

        mas32.neighborListSize = neighborListSize;
        CUDA_SAFE_CALL(cudaMemcpy(mas32.d_neighborListInit,
                                  tetMesh.neighborList.data(),
                                  neighborListSize * sizeof(unsigned int),
                                  cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(mas32.d_neighborStart,
                                  tetMesh.neighborStart.data(),
                                  (ipc.vertexNum - tetMesh.abd_vertexOffset)
                                      * sizeof(unsigned int),
                                  cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(mas32.d_neighborNumInit,
                                  tetMesh.neighborNum.data(),
                                  (ipc.vertexNum - tetMesh.abd_vertexOffset)
                                      * sizeof(unsigned int),
                                  cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(mas32.d_partId_map_real,
                                  tetMesh.partId_map_real.data(),
                                  tetMesh.part_offset * GPU_MAS_BANKSIZE * sizeof(int),
                                  cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(mas32.d_real_map_partId,
                                  tetMesh.real_map_partId.data(),
                                  tetMesh.real_map_partId.size() * sizeof(int),
                                  cudaMemcpyHostToDevice));
        mas32.initPreconditioner_Matrix();
    }

    CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.rest_vertexes,
                              d_tetMesh.o_vertexes,
                              ipc.vertexNum * sizeof(double3),
                              cudaMemcpyDeviceToDevice));


#ifdef USE_QUADRATIC_BENDING
    // Precompute Q matrices for quadratic bending
    if(tetMesh.tri_edges.size() > 0)
    {
        printf("Precomputing Q matrices for quadratic bending (%zu edges)...\n",
               tetMesh.tri_edges.size());

        // Allocate host memory for Q matrices
        std::vector<Eigen::Matrix4d> Q_host(tetMesh.tri_edges.size());

        // Download rest positions, edges, and adjacency from device
        std::vector<double3> rest_verts_host(ipc.vertexNum);
        CUDA_SAFE_CALL(cudaMemcpy(rest_verts_host.data(),
                                  d_tetMesh.rest_vertexes,
                                  ipc.vertexNum * sizeof(double3),
                                  cudaMemcpyDeviceToHost));

        // Call precomputation function (defined in femEnergy.cu)
        PrepareQuadBendingQ(rest_verts_host.data(),
                            tetMesh.tri_edges.data(),
                            tetMesh.tri_edges_adj_points.data(),  // CPU端叫tri_edges_adj_points
                            tetMesh.tri_edges.size(),
                            Q_host.data());

        // Upload Q matrices to device
        CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.quad_bending_Q,
                                  Q_host.data(),
                                  tetMesh.tri_edges.size() * sizeof(Eigen::Matrix4d),
                                  cudaMemcpyHostToDevice));

        printf("Quadratic bending Q matrices uploaded successfully.\n");

        // Optional: Print first Q matrix for verification
        if(tetMesh.tri_edges.size() > 0)
        {
            printf("First Q matrix:\n");
            for(int i = 0; i < 4; i++)
            {
                printf("  [%10.6f %10.6f %10.6f %10.6f]\n",
                       Q_host[0](i, 0),
                       Q_host[0](i, 1),
                       Q_host[0](i, 2),
                       Q_host[0](i, 3));
            }

            // Check for NaN or Inf
            int nan_count = 0;
            int inf_count = 0;
            for(size_t i = 0; i < Q_host.size(); i++)
            {
                for(int r = 0; r < 4; r++)
                {
                    for(int c = 0; c < 4; c++)
                    {
                        if(std::isnan(Q_host[i](r, c)))
                            nan_count++;
                        if(std::isinf(Q_host[i](r, c)))
                            inf_count++;
                    }
                }
            }
            if(nan_count > 0 || inf_count > 0)
            {
                printf("WARNING: Q matrices contain %d NaN values and %d Inf values!\n",
                       nan_count,
                       inf_count);
            }
        }
    }
#endif


    ipc.buildBVH();
    ipc.init(tetMesh.meanMass, tetMesh.meanVolum, tetMesh.minConer, tetMesh.maxConer, linear_system_buff_scale);
    ipc.set_paper_current_newton_stop(runtime_options.newton_stop == "paper-current");
    const bool agipc_core_enabled=runtime_options.solver==gipc::SolverMode::AGIPC;
    if(runtime_options.agipc_diagnostics || agipc_core_enabled)
    {
        std::string fallback_diagnostics_directory;
        std::string coarse_diagnostics_directory;
        if(runtime_options.agipc_diagnostics)
        {
            const std::filesystem::path diagnostic_base=
                runtime_options.metrics_path.empty()
                    ? std::filesystem::path(gipc::output_dir())/"agipc_diagnostics.json"
                    : std::filesystem::path(runtime_options.metrics_path);
            fallback_diagnostics_directory=(diagnostic_base.parent_path()
                /(diagnostic_base.stem().string()+"_fallback_samples")).string();
            coarse_diagnostics_directory=(diagnostic_base.parent_path()
                /(diagnostic_base.stem().string()+"_coarse_samples")).string();
        }
        agipc::configure_galerkin(runtime_options.agipc_fine_correction_iterations,
                                  agipc_core_enabled,
                                  std::move(fallback_diagnostics_directory),
                                  std::move(coarse_diagnostics_directory),
                                  runtime_options.agipc_coarse_preconditioner == "mas32",
                                  runtime_options.agipc_mas_validation,runtime_options.agipc_mas_reuse);
        agipc::initialize_criterion(tetMesh,runtime_options.agipc_threshold,
                                    runtime_options.agipc_max_levels);
    }

    printf("bboxDiagSize2: %f\n", ipc.bboxDiagSize2);
    printf("maxConer: %f  %f   %f           minCorner: %f  %f   %f\n",
           tetMesh.maxConer.x,
           tetMesh.maxConer.y,
           tetMesh.maxConer.z,
           tetMesh.minConer.x,
           tetMesh.minConer.y,
           tetMesh.minConer.z);

    printf("restSNKE: %f\n", ipc.RestNHEnergy);
    ipc.buildCP();

    ipc._moveDir          = ipc.pcg_data.dx;
    ipc.animation_subRate = 1.0 / motion_rate;
    //ipc.animation_fullRate = ipc.animation_subRate;
    ipc.computeXTilta(d_tetMesh, 1);
    ///////////////////////////////////////////////////////////////////////////////////

    ipc.create_LinearSystem(d_tetMesh);

    bvs.resize(2 * ipc.edge_Num - 1);
    nodes.resize(2 * ipc.edge_Num - 1);
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());
    CUDA_SAFE_CALL(cudaMemcpy(
        &bvs[0], ipc.bvh_e._bvs, (2 * ipc.edge_Num - 1) * sizeof(AABB), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(
        &nodes[0], ipc.bvh_e._nodes, (2 * ipc.edge_Num - 1) * sizeof(Node), cudaMemcpyDeviceToHost));
}


void outputAnimationMeshInfo(std::string pathCloth, std::string pathBody)
{
    std::stringstream ss;
    ss << pathCloth;
    ss.fill('0');
    ss.width(5);
    ss << (surfNumId) / 1;  // / 10;
    //if (surfNumId % 10 != 0) return;
    ss << ".obj";
    std::string file_path = ss.str();
    std::ofstream    outSurf(file_path);

    std::map<int, int> meshToSurf;
    for(int i = 0; i < bodyVertOffset; i++)
    {
        const auto& pos = tetMesh.vertexes[i];
        outSurf << "v " << pos.x << " " << pos.y << " " << pos.z << std::endl;
        //meshToSurf[tetMesh.surfVerts[i]] = i;
    }

    for(int i = 0; i < tetMesh.triangles.size(); i++)
    {
        const auto& tri = tetMesh.triangles[i];
        outSurf << "f " << tri.x + 1 << " " << tri.y + 1 << " " << tri.z + 1 << std::endl;
    }
    outSurf.close();

    std::stringstream ss2;
    ss2 << pathBody;
    ss2.fill('0');
    ss2.width(5);
    ss2 << (surfNumId) / 1;  // / 10;
    //if (surfNumId % 10 != 0) return;
    ss2 << ".obj";
    std::string file_path2 = ss2.str();
    std::ofstream    outSurf2(file_path2);

    //std::map<int, int> meshToSurf;
    for(int i = bodyVertOffset; i < tetMesh.vertexes.size(); i++)
    {
        const auto& pos = tetMesh.vertexes[i];
        outSurf2 << "v " << pos.x << " " << pos.y << " " << pos.z << std::endl;
        //meshToSurf[tetMesh.surfVerts[i]] = i;
    }

    for(int i = 0; i < clothFaceOffset; i++)
    {
        const auto& tri = tetMesh.surface[i];
        outSurf2 << "f " << tri.x + 1 - bodyVertOffset << " " << tri.y + 1 - bodyVertOffset
                 << " " << tri.z + 1 - bodyVertOffset << std::endl;
    }
    outSurf2.close();
    surfNumId++;
}
bool pri = true;
void display(void)
{
    draw_Scene3D();
    std::filesystem::exists(std::string{gipc::output_dir()})
        || std::filesystem::create_directory(std::string{gipc::output_dir()});
    auto output_path = std::string{gipc::output_dir()} + "saveSurface/";

    std::filesystem::exists(output_path) || std::filesystem::create_directory(output_path);

    //if(stop)
    //    return;


    ipc.IPC_Solver(d_tetMesh);

    if(ipc.animation && true)
    {
        std::string filename =
            "triMesh/body4/postcvpr_big_body_" + std::to_string(frameId + 1) + ".obj";
        frameId++;
        tetMesh.load_animation(filename, 1, make_double3(-1, -0.5, -0.5));
        CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.targetVert,
                                  tetMesh.targetPos.data(),
                                  tetMesh.softNum * sizeof(double3),
                                  cudaMemcpyHostToDevice));
    }


    CUDA_SAFE_CALL(cudaMemcpy(
        &bvs[0], ipc.bvh_e._bvs, (2 * ipc.edge_Num - 1) * sizeof(AABB), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(
        &nodes[0], ipc.bvh_e._nodes, (2 * ipc.edge_Num - 1) * sizeof(Node), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(tetMesh.vertexes.data(),
                              ipc._vertexes,
                              ipc.vertexNum * sizeof(double3),
                              cudaMemcpyDeviceToHost));


    if(screenshot)
    {
        std::stringstream ss;
        ss << "saveScreen/step_";
        ss.fill('0');
        ss.width(5);
        ss << step / 1;
        std::string file_path = ss.str();
        SaveScreenShot(window_width, window_height, file_path);
    }
    step++;
    printf("step:  %d\n", step);

    //if(step >= 160)
    //{
    //    std::cout << "step: " << step << " finished." << std::endl;
    //    exit(0);
    //}
}

void init(void)
{
    Init_CUDA();

    //main2();

    GLenum err = glewInit();
    if(GLEW_OK != err)
    {
        /* Problem: glewInit failed, something is seriously wrong. */
        std::cerr << "Error: " << glewGetErrorString(err) << std::endl;
    }
    std::cerr << "Status: Using GLEW " << glewGetString(GLEW_VERSION) << std::endl;
    glClearColor(0.0, 0.0, 0.0, 1.0);


    LoadSettings();

    ipc.build_gipc_system(d_tetMesh);
    ipc.set_spmv_mode(runtime_options.spmv);
    ipc.set_frozen_linear_diagnostics_path(
        runtime_options.frozen_linear_diagnostics_path);

    initScene();

    if(!isSetShader)
    {
        glViewport(0, 0, window_width, window_height);
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        gluPerspective(45.0, (float)window_width / window_height, 10.1f, 500.0);
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        glTranslatef(0.0f, 0.0f, -3.0f);
    }
    else
    {
        glGenBuffers(1, &PN_vbo_);
        glGenVertexArrays(1, &VAO);
    }
    //glEnable(GL_DEPTH_TEST);
}

void init_headless()
{
    std::cerr << "[headless] init_cuda" << std::endl;
    Init_CUDA();
    std::cerr << "[headless] load_settings" << std::endl;
    LoadSettings();
    ipc.IPC_dt = runtime_options.dt;
    std::cerr << "[headless] build_system" << std::endl;
    ipc.build_gipc_system(d_tetMesh);
    set_gipc_merged_timing_breakdown_enabled(false);
    ipc.set_spmv_mode(runtime_options.spmv);
    ipc.set_frozen_linear_diagnostics_path(
        runtime_options.frozen_linear_diagnostics_path);
    std::cerr << "[headless] init_scene" << std::endl;
    initScene();
    std::cerr << "[headless] ready" << std::endl;
}

extern double totalTime;
extern double total_Cg_count;
extern int    totalNT;
extern int    total_Frames;
extern double totalCollisionPairs;
extern double maxCOllisionPairNum;

int run_headless()
{
    const auto wall_begin = std::chrono::steady_clock::now();
    init_headless();
    const auto initial_vertices = tetMesh.vertexes;
    const auto simulation_begin = std::chrono::steady_clock::now();

    for(int frame = 0; frame < runtime_options.frames; ++frame)
    {
        ipc.IPC_Solver(d_tetMesh);
        // A frozen-linear run intentionally stops before CCD, line search,
        // position update, and frame accounting. Do not enter the ordinary
        // headless metrics path with that deliberately incomplete frame.
        if(ipc.frozen_linear_diagnostics_complete())
            return 0;
    }

    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    CUDA_SAFE_CALL(cudaMemcpy(tetMesh.vertexes.data(),
                              ipc._vertexes,
                              ipc.vertexNum * sizeof(double3),
                              cudaMemcpyDeviceToHost));

    bool   finite_vertices = true;
    double min_y           = std::numeric_limits<double>::infinity();
    for(const auto& vertex : tetMesh.vertexes)
    {
        finite_vertices = finite_vertices && std::isfinite(vertex.x)
                          && std::isfinite(vertex.y) && std::isfinite(vertex.z);
        min_y = std::min(min_y, vertex.y);
    }
    double fem_min_y=std::numeric_limits<double>::infinity();
    double fem_max_y=-std::numeric_limits<double>::infinity();
    double fem_displacement2_sum=0.0;
    double fem_max_displacement2=0.0;
    double abd_max_displacement2=0.0;
    double3 fem_position_sum=make_double3(0,0,0);
    const size_t fem_offset=tetMesh.abd_vertexOffset;
    for(size_t i=0;i<tetMesh.vertexes.size();++i)
    {
        const auto& current=tetMesh.vertexes[i];
        const auto& initial=initial_vertices[i];
        const double dx=current.x-initial.x;
        const double dy=current.y-initial.y;
        const double dz=current.z-initial.z;
        const double displacement2=dx*dx+dy*dy+dz*dz;
        if(i<fem_offset)
            abd_max_displacement2=std::max(abd_max_displacement2,displacement2);
        else
        {
            fem_min_y=std::min(fem_min_y,current.y);
            fem_max_y=std::max(fem_max_y,current.y);
            fem_displacement2_sum+=displacement2;
            fem_max_displacement2=std::max(fem_max_displacement2,displacement2);
            fem_position_sum.x+=current.x;
            fem_position_sum.y+=current.y;
            fem_position_sum.z+=current.z;
        }
    }
    const size_t fem_vertex_count=tetMesh.vertexes.size()-fem_offset;

    const auto simulation_end = std::chrono::steady_clock::now();
    const double wall_time_ms =
        std::chrono::duration<double, std::milli>(simulation_end - wall_begin).count();
    const double simulation_wall_time_ms =
        std::chrono::duration<double, std::milli>(simulation_end - simulation_begin).count();

    gipc::Json metrics;
    metrics["scene"]            = runtime_options.scene;
    metrics["solver"]           = runtime_options.solver == gipc::SolverMode::AGIPC
                                      ? "agipc-core"
                                      : "stiffgipc";
    metrics["newton_stop"]      = runtime_options.newton_stop;
    metrics["newton_stop_effective"] =
        runtime_options.newton_stop == "paper-current"
            || runtime_options.solver == gipc::SolverMode::AGIPC
            ? "paper-current" : "legacy-previous";
    metrics["tet_mesh"]         = runtime_options.tet_mesh;
    const bool agipc_active=runtime_options.agipc_diagnostics
                            || runtime_options.solver==gipc::SolverMode::AGIPC;
    metrics["agipc_criterion_enabled"] = agipc_active;
    if(agipc_active)
        metrics["agipc_criterion_threshold"] = runtime_options.agipc_threshold;
    if(agipc_active)
        metrics["agipc_criterion"] = agipc::criterion_summary();
    metrics["cloth_mesh"]       = runtime_options.cloth_mesh;
    metrics["framework"]        = runtime_options.framework;
    metrics["preconditioner"]   = runtime_options.preconditioner;
    metrics["agipc_coarse_preconditioner"] = runtime_options.agipc_coarse_preconditioner;
    metrics["agipc_mas_validation"]=runtime_options.agipc_mas_validation;
    metrics["agipc_mas_reuse"]=runtime_options.agipc_mas_reuse;
    metrics["spmv"]             = gipc::to_string(runtime_options.spmv);
    const bool paper_mixed_scene = runtime_options.scene == "paper-fig12-coupling-scaled"
                                   || runtime_options.scene == "paper-fig15-cloth-abd-scaled";
    metrics["body_mode"]        = paper_mixed_scene
                                        ? runtime_options.body_mode
                                        : "scene-defined";
    metrics["frames_requested"] = runtime_options.frames;
    metrics["frames_completed"] = total_Frames;
    metrics["total_vertices"]   = ipc.vertexNum;
    metrics["abd_vertices"]     = tetMesh.abd_vertexOffset;
    metrics["fem_vertices"]     = ipc.vertexNum - tetMesh.abd_vertexOffset;
    metrics["total_tetrahedra"] = ipc.tetrahedraNum;
    metrics["abd_tetrahedra"]   = tetMesh.abd_tetOffset;
    metrics["fem_tetrahedra"]   = ipc.tetrahedraNum - tetMesh.abd_tetOffset;
    metrics["surface_faces"]     = ipc.surface_Num;
    metrics["abd_body_count"]   = ipc.abd_fem_count_info.abd_body_num;
    metrics["fem_body_count"]   = ipc.abd_fem_count_info.fem_body_num;
    const auto full_fem_dof_reference = 3 * ipc.vertexNum;
    const auto linear_system_dof =
        12 * static_cast<int>(ipc.abd_fem_count_info.abd_body_num)
        + 3 * (ipc.vertexNum - tetMesh.abd_vertexOffset);
    metrics["full_fem_dof_reference"] = full_fem_dof_reference;
    metrics["linear_system_dof"]      = linear_system_dof;
    metrics["dof_reduction_ratio"] = linear_system_dof > 0
                                           ? static_cast<double>(full_fem_dof_reference)
                                                 / linear_system_dof
                                           : 0.0;
    if(runtime_options.scene == "paper-fig12-coupling-scaled")
    {
        metrics["scene_scale"]     = "REDUCED_SCALE";
        metrics["figure12_bunny_count"] = runtime_options.figure12_bunny_count;
        metrics["figure12_collision_buffer_scale"] =
            runtime_options.figure12_collision_buffer_scale;
        metrics["figure12_linear_system_buffer_scale"] =
            runtime_options.figure12_linear_system_buffer_scale;
        metrics["cloth_vertices"]  = figure12_cloth_vertex_count;
        metrics["cloth_triangles"] = figure12_cloth_triangle_count;
        metrics["primary_bunny_young_modulus"] =
            figure12_primary_bunny_young_modulus;
        metrics["secondary_bunny_young_modulus"] =
            figure12_secondary_bunny_young_modulus;
    }
    else if(runtime_options.scene == "paper-fig15-cloth-abd-scaled")
    {
        metrics["scene_scale"] = "REDUCED_SCALE";
        metrics["figure15_obstacle"] =
            "tetMesh/wrecking-ball-mesh/wrecking-ball/ball.obj_.msh";
        metrics["figure15_obstacle_fixed"] = true;
        metrics["cloth_vertices"] = figure15_cloth_vertex_count;
        metrics["cloth_triangles"] = figure15_cloth_triangle_count;
        metrics["cloth_young_modulus"] = ipc.clothYoungModulus;
    }
    metrics["young_modulus"]    = runtime_options.young_modulus;
    metrics["dt"]               = runtime_options.dt;
    metrics["wall_time_ms"]     = wall_time_ms;
    metrics["simulation_wall_time_ms"] = simulation_wall_time_ms;
    metrics["simulation_time_ms"] = totalTime;
    metrics["newton_iterations"]  = totalNT;
    metrics["pcg_iterations"]     = total_Cg_count;
    metrics["cumulative_self_collision_pair_samples"] = totalCollisionPairs;
    metrics["maximum_self_collision_pairs"] = maxCOllisionPairNum;
    metrics["last_self_collision_pairs"] = ipc.h_cpNum[0];
    metrics["last_ground_collision_pairs"] = ipc.h_gpNum;
    metrics["average_newton_per_frame"] = total_Frames > 0
                                                ? static_cast<double>(totalNT) / total_Frames
                                                : 0.0;
    metrics["average_pcg_per_newton"] = totalNT > 0
                                              ? total_Cg_count / totalNT
                                              : 0.0;
    metrics["average_simulation_ms_per_frame"] = total_Frames > 0
                                                       ? totalTime / total_Frames
                                                       : 0.0;
    metrics["finite_vertices"]    = finite_vertices;
    metrics["minimum_y"]          = min_y;
    metrics["ground_penetration"] = std::max(0.0, -1.0 - min_y);
    if(fem_vertex_count>0)
    {
        metrics["fem_minimum_y"] = fem_min_y;
        metrics["fem_maximum_y"] = fem_max_y;
        metrics["fem_rms_displacement"] =
            std::sqrt(fem_displacement2_sum/static_cast<double>(fem_vertex_count));
        metrics["fem_maximum_displacement"] = std::sqrt(fem_max_displacement2);
        metrics["fem_position_mean"] = {
            fem_position_sum.x/fem_vertex_count,
            fem_position_sum.y/fem_vertex_count,
            fem_position_sum.z/fem_vertex_count};
    }
    metrics["abd_maximum_displacement"] = std::sqrt(abd_max_displacement2);
    if(!runtime_options.fem_final_state_path.empty())
    {
        const std::filesystem::path final_state_path(runtime_options.fem_final_state_path);
        if(!final_state_path.parent_path().empty())
            std::filesystem::create_directories(final_state_path.parent_path());
        std::ofstream output(final_state_path);
        output.precision(17);
        output << "vertex,x,y,z\n";
        for(size_t i=fem_offset;i<tetMesh.vertexes.size();++i)
        {
            const auto& position=tetMesh.vertexes[i];
            output << i-fem_offset << ',' << position.x << ',' << position.y << ','
                   << position.z << '\n';
        }
        if(!output)
        {
            std::cerr << "failed to write FEM final state: " << final_state_path << '\n';
            return 3;
        }
        metrics["fem_final_state_path"] = final_state_path.string();
    }

    if(!runtime_options.metrics_path.empty())
    {
        const std::filesystem::path metrics_path(runtime_options.metrics_path);
        if(!metrics_path.parent_path().empty())
            std::filesystem::create_directories(metrics_path.parent_path());
        std::ofstream output(metrics_path);
        output << metrics.dump(2,
                               ' ',
                               false,
                               gipc::Json::error_handler_t::replace)
               << '\n';
        if(!output)
        {
            std::cerr << "failed to write metrics: " << metrics_path << '\n';
            return 3;
        }
    }
    std::cout << metrics.dump(-1,
                              ' ',
                              false,
                              gipc::Json::error_handler_t::replace)
              << '\n';
    return finite_vertices ? 0 : 3;
}


void idle_func()
{
    glutPostRedisplay();
}

void reshape_func(GLint width, GLint height)
{
    //window_width = width;
    //window_height = height;

    glViewport(0, 0, width, height);
    if(!isSetShader)
    {
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();

        gluPerspective(45.0, (float)width / height, 0.1, 500.0);

        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        glTranslatef(0.0f, 0.0f, -3.0f);
    }
    //glTranslatef(0.5f, 0.5f, -4.0f);
}

void keyboard_func(unsigned char key, int x, int y)
{
    if(key == 'w')
    {
        zTrans += .3f;
    }

    if(key == 's')
    {
        zTrans -= .3f;
    }

    if(key == 'a')
    {
        xTrans += .3f;
    }

    if(key == 'd')
    {
        xTrans -= .3f;
    }

    if(key == 'q')
    {
        yTrans -= .3f;
    }

    if(key == 'e')
    {
        yTrans += .3f;
    }

    if(key == '/')
    {
        screenshot = !screenshot;
    }

    if(key == '9')
    {
        saveSurface = !saveSurface;
    }

    if(key == 'k')
    {
        drawSurface = !drawSurface;
    }

    if(key == 'f')
    {
        drawbvh = !drawbvh;
    }

    if(key == ' ')
    {
        stop = !stop;
    }
    glutPostRedisplay();
}

void special_keyboard_func(int key, int x, int y)
{
    glutPostRedisplay();
}

void mouse_func(int button, int state, int x, int y)
{
    if(state == GLUT_DOWN)
    {
        buttonState = 1;
    }
    else if(state == GLUT_UP)
    {
        buttonState = 0;
    }

    ox = x;
    oy = y;

    glutPostRedisplay();
}

void motion_func(int x, int y)
{
    float dx, dy;
    dx = (float)(x - ox);
    dy = (float)(y - oy);

    if(buttonState == 1)
    {
        xRot += dy / 5.0f;
        yRot += dx / 5.0f;
    }

    ox = x;
    oy = y;

    glutPostRedisplay();
}


void SpecialKey(GLint key, GLint x, GLint y)
{
    if(key == GLUT_KEY_DOWN)
    {
        change = true;
        initPath -= 1;
        if(initPath < 0)
        {
            initPath = obj_pathes.size() - 1;
        }
    }

    if(key == GLUT_KEY_UP)
    {
        change = true;
        initPath += 1;
        if(initPath == obj_pathes.size())
        {
            initPath = 0;
        }
    }
    glutPostRedisplay();
}


int main(int argc, char** argv)
{
    const auto parsed_options = gipc::parse_runtime_options(argc, argv);
    if(!parsed_options.ok)
    {
        std::cerr << parsed_options.error << '\n' << parsed_options.help;
        return parsed_options.exit_code;
    }
    if(parsed_options.show_help)
    {
        std::cout << parsed_options.help;
        return 0;
    }
    runtime_options = parsed_options.options;
    if(runtime_options.solver == gipc::SolverMode::AGIPCSymHessian
       || runtime_options.solver == gipc::SolverMode::AGIPCPaper)
    {
        std::cerr << "Requested AGIPC solver unavailable: symmetric Hessian and paper BVH stages are not implemented.\n";
        return 2;
    }
    if(!runtime_options.agipc_coarse_replay_path.empty())
    {
        Init_CUDA();
        try
        {
            const auto result=agipc::replay_coarse_snapshot(
                runtime_options.agipc_coarse_replay_path);
            if(!runtime_options.metrics_path.empty())
            {
                std::ofstream output;
                output.exceptions(std::ios::badbit|std::ios::failbit);
                const std::filesystem::path path(runtime_options.metrics_path);
                if(!path.parent_path().empty())
                    std::filesystem::create_directories(path.parent_path());
                output.open(path);
                output<<result.dump(2)<<'\n';
                output.close();
            }
            std::cout<<result.dump(2)<<'\n';
            return result.value("passed",false)?0:3;
        }
        catch(const std::exception& error)
        {
            std::cerr<<"Coarse replay failed: "<<error.what()<<'\n';
            return 3;
        }
    }
    if(runtime_options.agipc_self_test)
    {
        Init_CUDA();
        try
        {
            gipc::Json result;
            result["criterion"] = agipc::criterion_self_test();
            result["mapping"] = agipc::mapping_self_test();
            result["galerkin"] = agipc::galerkin_self_test();
            result["passed"] = result["criterion"]["passed"].get<bool>()
                               && result["mapping"]["passed"].get<bool>()
                               && result["galerkin"]["passed"].get<bool>();
            std::cout << result.dump(2) << '\n';
        }
        catch(const std::exception& error)
        {
            std::cerr << "AGIPC self-test failed: " << error.what() << '\n';
            return 3;
        }
        return 0;
    }
    if(runtime_options.spmv_self_test)
    {
        Init_CUDA();
        const auto result = gipc::run_spmv_equivalence_self_test();
        gipc::Json metrics;
        metrics["test"] = "spmv_equivalence";
        metrics["passed"] = result.passed;
        metrics["block_rows"] = result.block_rows;
        metrics["triplet_count"] = result.triplet_count;
        metrics["legacy_reference_max_abs_error"] =
            result.legacy_reference_max_abs_error;
        metrics["srbk_reference_max_abs_error"] =
            result.srbk_reference_max_abs_error;
        metrics["legacy_srbk_max_abs_error"] = result.legacy_srbk_max_abs_error;
        metrics["legacy_srbk_max_relative_error"] =
            result.legacy_srbk_max_relative_error;
        metrics["hybrid8_reference_max_abs_error"] =
            result.hybrid8_reference_max_abs_error;
        metrics["hybrid16_reference_max_abs_error"] =
            result.hybrid16_reference_max_abs_error;
        std::cout << metrics.dump(2) << '\n';
        return result.passed ? 0 : 3;
    }
    if(runtime_options.mas32_self_test)
    {
        Init_CUDA();
        const auto result = gpu_mas32::run_hierarchy_self_test();
        gipc::Json metrics;
        metrics["test"] = "traditional_mas32_hierarchy";
        metrics["passed"] = result.passed;
        metrics["valid_nodes"] = result.valid_nodes;
        metrics["padded_nodes"] = result.padded_nodes;
        metrics["expected_components"] = result.expected_components;
        metrics["actual_components"] = result.actual_components;
        metrics["fine_mask_mismatches"] = result.fine_mask_mismatches;
        metrics["coarse_mapping_mismatches"] = result.coarse_mapping_mismatches;
        metrics["going_next_mismatches"] = result.going_next_mismatches;
        metrics["local_gpu_checks"]=gpu_mas32::run_local_diagnostics_gpu_self_test();
        metrics["passed"]=result.passed && metrics["local_gpu_checks"].value("passed",false);
        std::cout << metrics.dump(2) << '\n';
        return metrics.value("passed",false) ? 0 : 3;
    }
    if(!std::filesystem::is_directory(assets_dir))
    {
        const auto local_assets = std::filesystem::current_path() / "Assets";
        if(std::filesystem::is_directory(local_assets))
        {
            assets_dir = local_assets.string() + "/";
            metis_dir  = assets_dir + "sorted_mesh/";
        }
    }
    if(runtime_options.scene == "stiff-bunny-drop")
    {
        std::filesystem::path mesh_path = runtime_options.tet_mesh.empty()
                                              ? std::filesystem::path(assets_dir)
                                                    / "tetMesh/bunny2.msh"
                                              : std::filesystem::path(runtime_options.tet_mesh);
        if(mesh_path.is_relative() && !std::filesystem::exists(mesh_path))
            mesh_path = std::filesystem::path(assets_dir) / "tetMesh" / mesh_path;
        if(mesh_path.extension() != ".msh" || !std::filesystem::is_regular_file(mesh_path))
        {
            std::cerr << "tet mesh must be an existing .msh file: " << mesh_path << '\n';
            return 2;
        }
        runtime_options.tet_mesh = mesh_path.string();
        if(runtime_options.frames == 0)
            runtime_options.frames = 30;
    }
    else if(runtime_options.scene == "paper-fig12-coupling-scaled"
            || runtime_options.scene == "paper-fig15-cloth-abd-scaled")
    {
        std::filesystem::path mesh_path = runtime_options.cloth_mesh.empty()
                                              ? std::filesystem::path(assets_dir)
                                                    / "triMesh/cloth_high.obj"
                                              : std::filesystem::path(runtime_options.cloth_mesh);
        if(mesh_path.is_relative() && !std::filesystem::exists(mesh_path))
            mesh_path = std::filesystem::path(assets_dir) / "triMesh" / mesh_path;
        if(mesh_path.extension() != ".obj" || !std::filesystem::is_regular_file(mesh_path))
        {
            std::cerr << "cloth mesh must be an existing .obj file: " << mesh_path << '\n';
            return 2;
        }
        runtime_options.cloth_mesh = mesh_path.string();
        if(runtime_options.scene == "paper-fig15-cloth-abd-scaled"
           && runtime_options.frames == 0)
            runtime_options.frames = 40;
    }

    if(runtime_options.headless)
        return run_headless();

    glutInit(&argc, argv);
    //glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);

    glutSetOption(GLUT_MULTISAMPLE, 16);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH | GLUT_MULTISAMPLE);

    glutInitWindowSize(window_width, window_height);
    glutInitWindowPosition(0, 0);
    glutCreateWindow("FEM");

    init();

    glDepthMask(GL_TRUE);
    glEnable(GL_DEPTH_TEST);


    glEnable(GL_MULTISAMPLE);
    glHint(GL_MULTISAMPLE_FILTER_HINT_NV, GL_NICEST);


    glutDisplayFunc(display);


    //glutDisplayFunc(display_func);
    glutReshapeFunc(reshape_func);
    glutKeyboardFunc(keyboard_func);
    glutSpecialFunc(&SpecialKey);
    glutMouseFunc(mouse_func);
    glutMotionFunc(motion_func);
    glutIdleFunc(idle_func);


    glutMainLoop();
    //return 0;
}
