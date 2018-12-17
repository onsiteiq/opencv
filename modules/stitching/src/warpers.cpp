/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "precomp.hpp"
#include "opencl_kernels_stitching.hpp"

#include <algorithm>

namespace cv {
namespace detail {

void ProjectorBase::setCameraParams(InputArray _K, InputArray _R, InputArray _T)
{
    Mat K = _K.getMat(), R = _R.getMat(), T = _T.getMat();

    CV_Assert(K.size() == Size(3, 3) && K.type() == CV_32F);
    CV_Assert(R.size() == Size(3, 3) && R.type() == CV_32F);
    CV_Assert((T.size() == Size(1, 3) || T.size() == Size(3, 1)) && T.type() == CV_32F);

    Mat_<float> K_(K);
    k[0] = K_(0,0); k[1] = K_(0,1); k[2] = K_(0,2);
    k[3] = K_(1,0); k[4] = K_(1,1); k[5] = K_(1,2);
    k[6] = K_(2,0); k[7] = K_(2,1); k[8] = K_(2,2);

    Mat_<float> Rinv = R.t();
    rinv[0] = Rinv(0,0); rinv[1] = Rinv(0,1); rinv[2] = Rinv(0,2);
    rinv[3] = Rinv(1,0); rinv[4] = Rinv(1,1); rinv[5] = Rinv(1,2);
    rinv[6] = Rinv(2,0); rinv[7] = Rinv(2,1); rinv[8] = Rinv(2,2);

    Mat_<float> R_Kinv = R * K.inv();
    r_kinv[0] = R_Kinv(0,0); r_kinv[1] = R_Kinv(0,1); r_kinv[2] = R_Kinv(0,2);
    r_kinv[3] = R_Kinv(1,0); r_kinv[4] = R_Kinv(1,1); r_kinv[5] = R_Kinv(1,2);
    r_kinv[6] = R_Kinv(2,0); r_kinv[7] = R_Kinv(2,1); r_kinv[8] = R_Kinv(2,2);

    Mat_<float> K_Rinv = K * Rinv;
    k_rinv[0] = K_Rinv(0,0); k_rinv[1] = K_Rinv(0,1); k_rinv[2] = K_Rinv(0,2);
    k_rinv[3] = K_Rinv(1,0); k_rinv[4] = K_Rinv(1,1); k_rinv[5] = K_Rinv(1,2);
    k_rinv[6] = K_Rinv(2,0); k_rinv[7] = K_Rinv(2,1); k_rinv[8] = K_Rinv(2,2);

    Mat_<float> T_(T.reshape(0, 3));
    t[0] = T_(0,0); t[1] = T_(1,0); t[2] = T_(2,0);
}


Point2f PlaneWarper::warpPoint(const Point2f &pt, InputArray K, InputArray R, InputArray T)
{
    projector_.setCameraParams(K, R, T);
    Point2f uv;
    projector_.mapForward(pt.x, pt.y, uv.x, uv.y);
    return uv;
}

Point2f PlaneWarper::warpPoint(const Point2f &pt, InputArray K, InputArray R)
{
    float tz[] = {0.f, 0.f, 0.f};
    Mat_<float> T(3, 1, tz);
    return warpPoint(pt, K, R, T);
}

Rect PlaneWarper::buildMaps(Size src_size, InputArray K, InputArray R, OutputArray xmap, OutputArray ymap)
{
    return buildMaps(src_size, K, R, Mat::zeros(3, 1, CV_32FC1), xmap, ymap);
}

Rect PlaneWarper::buildMaps(Size src_size, InputArray K, InputArray R, InputArray T, OutputArray _xmap, OutputArray _ymap)
{
    projector_.setCameraParams(K, R, T);

    Point dst_tl, dst_br;
    detectResultRoi(src_size, dst_tl, dst_br);

    Size dsize(dst_br.x - dst_tl.x + 1, dst_br.y - dst_tl.y + 1);
    _xmap.create(dsize, CV_32FC1);
    _ymap.create(dsize, CV_32FC1);

#ifdef HAVE_OPENCL
    if (ocl::isOpenCLActivated())
    {
        ocl::Kernel k("buildWarpPlaneMaps", ocl::stitching::warpers_oclsrc);
        if (!k.empty())
        {
            int rowsPerWI = ocl::Device::getDefault().isIntel() ? 4 : 1;
            Mat k_rinv(1, 9, CV_32FC1, projector_.k_rinv), t(1, 3, CV_32FC1, projector_.t);
            UMat uxmap = _xmap.getUMat(), uymap = _ymap.getUMat(),
                    uk_rinv = k_rinv.getUMat(ACCESS_READ), ut = t.getUMat(ACCESS_READ);

            k.args(ocl::KernelArg::WriteOnlyNoSize(uxmap), ocl::KernelArg::WriteOnly(uymap),
                   ocl::KernelArg::PtrReadOnly(uk_rinv), ocl::KernelArg::PtrReadOnly(ut),
                   dst_tl.x, dst_tl.y, 1/projector_.scale, rowsPerWI);

            size_t globalsize[2] = { (size_t)dsize.width, ((size_t)dsize.height + rowsPerWI - 1) / rowsPerWI };
            if (k.run(2, globalsize, NULL, true))
            {
                CV_IMPL_ADD(CV_IMPL_OCL);
                return Rect(dst_tl, dst_br);
            }
        }
    }
#endif

    Mat xmap = _xmap.getMat(), ymap = _ymap.getMat();

    float x, y;
    for (int v = dst_tl.y; v <= dst_br.y; ++v)
    {
        for (int u = dst_tl.x; u <= dst_br.x; ++u)
        {
            projector_.mapBackward(static_cast<float>(u), static_cast<float>(v), x, y);
            xmap.at<float>(v - dst_tl.y, u - dst_tl.x) = x;
            ymap.at<float>(v - dst_tl.y, u - dst_tl.x) = y;
        }
    }

    return Rect(dst_tl, dst_br);
}


Point PlaneWarper::warp(InputArray src, InputArray K, InputArray R, InputArray T, int interp_mode, int border_mode,
                        OutputArray dst)
{
    UMat uxmap, uymap;
    Rect dst_roi = buildMaps(src.size(), K, R, T, uxmap, uymap);

    dst.create(dst_roi.height + 1, dst_roi.width + 1, src.type());
    remap(src, dst, uxmap, uymap, interp_mode, border_mode);

    return dst_roi.tl();
}

Point PlaneWarper::warp(InputArray src, InputArray K, InputArray R,
                        int interp_mode, int border_mode, OutputArray dst)
{
    float tz[] = {0.f, 0.f, 0.f};
    Mat_<float> T(3, 1, tz);
    return warp(src, K, R, T, interp_mode, border_mode, dst);
}

Rect PlaneWarper::warpRoi(Size src_size, InputArray K, InputArray R, InputArray T)
{
    projector_.setCameraParams(K, R, T);

    Point dst_tl, dst_br;
    detectResultRoi(src_size, dst_tl, dst_br);

    return Rect(dst_tl, Point(dst_br.x + 1, dst_br.y + 1));
}

Rect PlaneWarper::warpRoi(Size src_size, InputArray K, InputArray R)
{
    float tz[] = {0.f, 0.f, 0.f};
    Mat_<float> T(3, 1, tz);
    return warpRoi(src_size, K, R, T);
}


void PlaneWarper::detectResultRoi(Size src_size, Point &dst_tl, Point &dst_br)
{
    float tl_uf = std::numeric_limits<float>::max();
    float tl_vf = std::numeric_limits<float>::max();
    float br_uf = -std::numeric_limits<float>::max();
    float br_vf = -std::numeric_limits<float>::max();

    float u, v;

    projector_.mapForward(0, 0, u, v);
    tl_uf = std::min(tl_uf, u); tl_vf = std::min(tl_vf, v);
    br_uf = std::max(br_uf, u); br_vf = std::max(br_vf, v);

    projector_.mapForward(0, static_cast<float>(src_size.height - 1), u, v);
    tl_uf = std::min(tl_uf, u); tl_vf = std::min(tl_vf, v);
    br_uf = std::max(br_uf, u); br_vf = std::max(br_vf, v);

    projector_.mapForward(static_cast<float>(src_size.width - 1), 0, u, v);
    tl_uf = std::min(tl_uf, u); tl_vf = std::min(tl_vf, v);
    br_uf = std::max(br_uf, u); br_vf = std::max(br_vf, v);

    projector_.mapForward(static_cast<float>(src_size.width - 1), static_cast<float>(src_size.height - 1), u, v);
    tl_uf = std::min(tl_uf, u); tl_vf = std::min(tl_vf, v);
    br_uf = std::max(br_uf, u); br_vf = std::max(br_vf, v);

    dst_tl.x = static_cast<int>(tl_uf);
    dst_tl.y = static_cast<int>(tl_vf);
    dst_br.x = static_cast<int>(br_uf);
    dst_br.y = static_cast<int>(br_vf);
}


Point2f AffineWarper::warpPoint(const Point2f &pt, InputArray K, InputArray H)
{
    Mat R, T;
    getRTfromHomogeneous(H, R, T);
    return PlaneWarper::warpPoint(pt, K, R, T);
}


Rect AffineWarper::buildMaps(Size src_size, InputArray K, InputArray H, OutputArray xmap, OutputArray ymap)
{
    Mat R, T;
    getRTfromHomogeneous(H, R, T);
    return PlaneWarper::buildMaps(src_size, K, R, T, xmap, ymap);
}


Point AffineWarper::warp(InputArray src, InputArray K, InputArray H,
                         int interp_mode, int border_mode, OutputArray dst)
{
    Mat R, T;
    getRTfromHomogeneous(H, R, T);
    return PlaneWarper::warp(src, K, R, T, interp_mode, border_mode, dst);
}


Rect AffineWarper::warpRoi(Size src_size, InputArray K, InputArray H)
{
    Mat R, T;
    getRTfromHomogeneous(H, R, T);
    return PlaneWarper::warpRoi(src_size, K, R, T);
}


void AffineWarper::getRTfromHomogeneous(InputArray H_, Mat &R, Mat &T)
{
    Mat H = H_.getMat();
    CV_Assert(H.size() == Size(3, 3) && H.type() == CV_32F);

    T = Mat::zeros(3, 1, CV_32F);
    R = H.clone();

    T.at<float>(0,0) = R.at<float>(0,2);
    T.at<float>(1,0) = R.at<float>(1,2);
    R.at<float>(0,2) = 0.f;
    R.at<float>(1,2) = 0.f;

    // we want to compensate transform to fit into plane warper
    R = R.t();
    T = (R * T) * -1;
}


void SphericalWarper::detectResultRoi(Size src_size, Point &dst_tl, Point &dst_br)
{
    detectResultRoiByBorder(src_size, dst_tl, dst_br);

    float tl_uf = static_cast<float>(dst_tl.x);
    float tl_vf = static_cast<float>(dst_tl.y);
    float br_uf = static_cast<float>(dst_br.x);
    float br_vf = static_cast<float>(dst_br.y);

	// This is warping backward from u,v corresponding to (0,1,0) and (0,-1,0)
	// to check if the poles are within the camera frame. I didn't realize that
	// originally so I implemented and inefficient version in detectResultRoiByBorder.
	// So at some point we need to choose between the two methods...

    float x = projector_.rinv[1];
    float y = projector_.rinv[4];
    float z = projector_.rinv[7];
    if (y > 0.f)
    {
        float x_ = (projector_.k[0] * x + projector_.k[1] * y) / z + projector_.k[2];
        float y_ = projector_.k[4] * y / z + projector_.k[5];
        if (x_ > 0.f && x_ < src_size.width && y_ > 0.f && y_ < src_size.height)
        {
            tl_uf = std::min(tl_uf, 0.f); tl_vf = std::min(tl_vf, static_cast<float>(CV_PI * projector_.scale));
            br_uf = std::max(br_uf, 0.f); br_vf = std::max(br_vf, static_cast<float>(CV_PI * projector_.scale));

            if ( br_vf > projector_.scale*static_cast<float>(CV_PI)-1 )
            {
                br_vf = projector_.scale*static_cast<float>(CV_PI)-1;
            }
        }
    }

    x = projector_.rinv[1];
    y = -projector_.rinv[4];
    z = projector_.rinv[7];
    if (y > 0.f)
    {
        float x_ = (projector_.k[0] * x + projector_.k[1] * y) / z + projector_.k[2];
        float y_ = projector_.k[4] * y / z + projector_.k[5];
        if (x_ > 0.f && x_ < src_size.width && y_ > 0.f && y_ < src_size.height)
        {
            tl_uf = std::min(tl_uf, 0.f); tl_vf = std::min(tl_vf, static_cast<float>(0));
            br_uf = std::max(br_uf, 0.f); br_vf = std::max(br_vf, static_cast<float>(0));
        }
    }

    dst_tl.x = static_cast<int>(tl_uf);
    dst_tl.y = static_cast<int>(tl_vf);
    dst_br.x = static_cast<int>(br_uf);
    dst_br.y = static_cast<int>(br_vf);
}


void SphericalWarper::getNewTLandBR_U( float& tl_uf, float& br_uf, float uf )
{
    // If tl and br are uninitialized just set and return
    if ( tl_uf == std::numeric_limits<float>::max() )
    {
        tl_uf = std::min(tl_uf, uf);
        br_uf = std::max(br_uf, uf); 
        return;
    }

    // Otherwise check for crossing of the +/-PI boundary and treat as a special case
    if ( ((tl_uf < 0.0f && uf > 0.0f) || (tl_uf > 0.0f && uf < 0.0f)) && cosf(uf/projector_.scale) < 0.0f )
    {
        tl_uf = -projector_.scale*static_cast<float>(CV_PI);
        br_uf = projector_.scale*static_cast<float>(CV_PI)-1;
    }
    else
    {
        tl_uf = std::min(tl_uf, uf);
        br_uf = std::max(br_uf, uf); 
    }
}

void SphericalWarper::detectResultRoiByBorder(Size src_size, Point &dst_tl, Point &dst_br)
{
    float tl_uf = std::numeric_limits<float>::max();
    float tl_vf = std::numeric_limits<float>::max();
    float br_uf = -std::numeric_limits<float>::max();
    float br_vf = -std::numeric_limits<float>::max();

    // To test if one of the poles is within the spherically projected frustum we
    // collect the points of the border (u,v) and reproject to the x,z plane. The
    // points form a polygon. If the origin is within the polygon then one of
    // the poles is within the frustum.

    std::vector<Point2f> borderProjX0(src_size.width);
    std::vector<Point2f> borderProjXH(src_size.width);
    std::vector<Point2f> borderProj0Y(src_size.height);
    std::vector<Point2f> borderProjWY(src_size.height);

    float u, v, tmp;
    for (float x = 0; x < src_size.width; ++x)
    {
        projector_.mapForward(static_cast<float>(x), 0, u, v);

        projector_.mapBackwardCartesian( u, v, borderProjX0[x].x, tmp, borderProjX0[x].y);

        getNewTLandBR_U(tl_uf, br_uf, u);
        //tl_uf = std::min(tl_uf, u); 
        tl_vf = std::min(tl_vf, v);
        //br_uf = std::max(br_uf, u); 
        br_vf = std::max(br_vf, v);

        projector_.mapForward(static_cast<float>(x), static_cast<float>(src_size.height - 1), u, v);

        projector_.mapBackwardCartesian( u, v, borderProjXH[x].x, tmp, borderProjXH[x].y);

        getNewTLandBR_U(tl_uf, br_uf, u);
        //tl_uf = std::min(tl_uf, u);
        tl_vf = std::min(tl_vf, v);
        //br_uf = std::max(br_uf, u);
        br_vf = std::max(br_vf, v);
    }
    for (int y = 0; y < src_size.height; ++y)
    {
        projector_.mapForward(0, static_cast<float>(y), u, v);

        projector_.mapBackwardCartesian( u, v, borderProj0Y[y].x, tmp, borderProj0Y[y].y);

        getNewTLandBR_U(tl_uf, br_uf, u);
        //tl_uf = std::min(tl_uf, u); 
        tl_vf = std::min(tl_vf, v);
        //br_uf = std::max(br_uf, u); 
        br_vf = std::max(br_vf, v);

        projector_.mapForward(static_cast<float>(src_size.width - 1), static_cast<float>(y), u, v);

        projector_.mapBackwardCartesian( u, v, borderProjWY[y].x, tmp, borderProjWY[y].y);

        getNewTLandBR_U(tl_uf, br_uf, u);
        //tl_uf = std::min(tl_uf, u); 
        tl_vf = std::min(tl_vf, v);
        //br_uf = std::max(br_uf, u); 
        br_vf = std::max(br_vf, v);
    }

    // Combine the x,z reprojected border into a single polygon with consistent handedness.

    std::vector<Point2f> xzPolygon;

    xzPolygon.insert( std::end(xzPolygon), std::begin(borderProjX0), std::end(borderProjX0) );
    xzPolygon.insert( std::end(xzPolygon), std::begin(borderProjWY), std::end(borderProjWY) );

    std::reverse(borderProjXH.begin(),borderProjXH.end());
    xzPolygon.insert( std::end(xzPolygon), std::begin(borderProjXH), std::end(borderProjXH) );

    std::reverse(borderProj0Y.begin(),borderProj0Y.end());
    xzPolygon.insert( std::end(xzPolygon), std::begin(borderProj0Y), std::end(borderProj0Y) );

    if ( pointPolygonTest(xzPolygon, Point2f(0.0f,0.0f), false) > 0.0 )
    {
        // One of the poles is inside our polygon. Do a basic test to see which pole.
        // This skips some edge conditions. 
        if ( br_vf > 0.5f*projector_.scale*static_cast<float>(CV_PI) )
        {
            br_vf = projector_.scale*static_cast<float>(CV_PI)-1;
        }
        else
        {
            tl_vf = 0.0f;
        }
    }

    dst_tl.x = static_cast<int>(tl_uf);
    dst_tl.y = static_cast<int>(tl_vf);
    dst_br.x = static_cast<int>(br_uf);
    dst_br.y = static_cast<int>(br_vf);
}

void SphericalPortraitWarper::detectResultRoi(Size src_size, Point &dst_tl, Point &dst_br)
{
    detectResultRoiByBorder(src_size, dst_tl, dst_br);

    float tl_uf = static_cast<float>(dst_tl.x);
    float tl_vf = static_cast<float>(dst_tl.y);
    float br_uf = static_cast<float>(dst_br.x);
    float br_vf = static_cast<float>(dst_br.y);

    float x = projector_.rinv[0];
    float y = projector_.rinv[3];
    float z = projector_.rinv[6];
    if (y > 0.f)
    {
        float x_ = (projector_.k[0] * x + projector_.k[1] * y) / z + projector_.k[2];
        float y_ = projector_.k[4] * y / z + projector_.k[5];
        if (x_ > 0.f && x_ < src_size.width && y_ > 0.f && y_ < src_size.height)
        {
            tl_uf = std::min(tl_uf, 0.f); tl_vf = std::min(tl_vf, static_cast<float>(CV_PI * projector_.scale));
            br_uf = std::max(br_uf, 0.f); br_vf = std::max(br_vf, static_cast<float>(CV_PI * projector_.scale));
        }
    }

    x = projector_.rinv[0];
    y = -projector_.rinv[3];
    z = projector_.rinv[6];
    if (y > 0.f)
    {
        float x_ = (projector_.k[0] * x + projector_.k[1] * y) / z + projector_.k[2];
        float y_ = projector_.k[4] * y / z + projector_.k[5];
        if (x_ > 0.f && x_ < src_size.width && y_ > 0.f && y_ < src_size.height)
        {
            tl_uf = std::min(tl_uf, 0.f); tl_vf = std::min(tl_vf, static_cast<float>(0));
            br_uf = std::max(br_uf, 0.f); br_vf = std::max(br_vf, static_cast<float>(0));
        }
    }

    dst_tl.x = static_cast<int>(tl_uf);
    dst_tl.y = static_cast<int>(tl_vf);
    dst_br.x = static_cast<int>(br_uf);
    dst_br.y = static_cast<int>(br_vf);
}

/////////////////////////////////////////// SphericalWarper ////////////////////////////////////////

Rect SphericalWarper::buildMaps(Size src_size, InputArray K, InputArray R, OutputArray xmap, OutputArray ymap)
{
#ifdef HAVE_OPENCL
    if (ocl::isOpenCLActivated())
    {
        ocl::Kernel k("buildWarpSphericalMaps", ocl::stitching::warpers_oclsrc);
        if (!k.empty())
        {
            int rowsPerWI = ocl::Device::getDefault().isIntel() ? 4 : 1;
            projector_.setCameraParams(K, R);

            Point dst_tl, dst_br;
            detectResultRoi(src_size, dst_tl, dst_br);

            Size dsize(dst_br.x - dst_tl.x + 1, dst_br.y - dst_tl.y + 1);
            xmap.create(dsize, CV_32FC1);
            ymap.create(dsize, CV_32FC1);

            Mat k_rinv(1, 9, CV_32FC1, projector_.k_rinv);
            UMat uxmap = xmap.getUMat(), uymap = ymap.getUMat(), uk_rinv = k_rinv.getUMat(ACCESS_READ);

            k.args(ocl::KernelArg::WriteOnlyNoSize(uxmap), ocl::KernelArg::WriteOnly(uymap),
                   ocl::KernelArg::PtrReadOnly(uk_rinv), dst_tl.x, dst_tl.y, 1/projector_.scale, rowsPerWI);

            size_t globalsize[2] = { (size_t)dsize.width, ((size_t)dsize.height + rowsPerWI - 1) / rowsPerWI };
            if (k.run(2, globalsize, NULL, true))
            {
                CV_IMPL_ADD(CV_IMPL_OCL);
                return Rect(dst_tl, dst_br);
            }
        }
    }
#endif
    return RotationWarperBase<SphericalProjector>::buildMaps(src_size, K, R, xmap, ymap);
}

Point SphericalWarper::warp(InputArray src, InputArray K, InputArray R, int interp_mode, int border_mode, OutputArray dst)
{
    UMat uxmap, uymap;
    Rect dst_roi = buildMaps(src.size(), K, R, uxmap, uymap);

    dst.create(dst_roi.height + 1, dst_roi.width + 1, src.type());
    remap(src, dst, uxmap, uymap, interp_mode, border_mode);

    return dst_roi.tl();
}

/////////////////////////////////////////// CylindricalWarper ////////////////////////////////////////

Rect CylindricalWarper::buildMaps(Size src_size, InputArray K, InputArray R, OutputArray xmap, OutputArray ymap)
{
#ifdef HAVE_OPENCL
    if (ocl::isOpenCLActivated())
    {
        ocl::Kernel k("buildWarpCylindricalMaps", ocl::stitching::warpers_oclsrc);
        if (!k.empty())
        {
            int rowsPerWI = ocl::Device::getDefault().isIntel() ? 4 : 1;
            projector_.setCameraParams(K, R);

            Point dst_tl, dst_br;
            detectResultRoi(src_size, dst_tl, dst_br);

            Size dsize(dst_br.x - dst_tl.x + 1, dst_br.y - dst_tl.y + 1);
            xmap.create(dsize, CV_32FC1);
            ymap.create(dsize, CV_32FC1);

            Mat k_rinv(1, 9, CV_32FC1, projector_.k_rinv);
            UMat uxmap = xmap.getUMat(), uymap = ymap.getUMat(), uk_rinv = k_rinv.getUMat(ACCESS_READ);

            k.args(ocl::KernelArg::WriteOnlyNoSize(uxmap), ocl::KernelArg::WriteOnly(uymap),
                   ocl::KernelArg::PtrReadOnly(uk_rinv), dst_tl.x, dst_tl.y, 1/projector_.scale,
                   rowsPerWI);

            size_t globalsize[2] = { (size_t)dsize.width, ((size_t)dsize.height + rowsPerWI - 1) / rowsPerWI };
            if (k.run(2, globalsize, NULL, true))
            {
                CV_IMPL_ADD(CV_IMPL_OCL);
                return Rect(dst_tl, dst_br);
            }
        }
    }
#endif
    return RotationWarperBase<CylindricalProjector>::buildMaps(src_size, K, R, xmap, ymap);
}

Point CylindricalWarper::warp(InputArray src, InputArray K, InputArray R, int interp_mode, int border_mode, OutputArray dst)
{
    UMat uxmap, uymap;
    Rect dst_roi = buildMaps(src.size(), K, R, uxmap, uymap);

    dst.create(dst_roi.height + 1, dst_roi.width + 1, src.type());
    remap(src, dst, uxmap, uymap, interp_mode, border_mode);

    return dst_roi.tl();
}

} // namespace detail
} // namespace cv
