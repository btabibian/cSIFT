/*
  This license applies to all parts except parts related to SIFT algorithm which is under its ownn license.
 
    This file is part of Implementation of Parallel edition of SIFT Algorithm

   Parallel SIFT is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Implementation of ARFL is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Implementation of ARFL.  If not, see <http://www.gnu.org/licenses/>.
  
    University of Edinburgh, hereby disclaims all copyright interest in the program 
    "Parallel SIFT" written by Behzad Tabibian.
----------
SIFT LICENSE CONDITIONS

Copyright (2005), University of British Columbia.


This software for the detection of invariant keypoints is being made

available for individual research use only.  Any commercial use or any

redistribution of this software requires a license from the University

of British Columbia.



The following patent has been issued for methods embodied in this

software: "Method and apparatus for identifying scale invariant

features in an image and use of same for locating an object in an

image," David G. Lowe, US Patent 6,711,293 (March 23,

2004). Provisional application filed March 8, 1999. Asignee: The

University of British Columbia.



For further details on obtaining a commercial license, contact David

Lowe (lowe@cs.ubc.ca) or the University-Industry Liaison Office of the

University of British Columbia. 



THE UNIVERSITY OF BRITISH COLUMBIA MAKES NO REPRESENTATIONS OR

WARRANTIES OF ANY KIND CONCERNING THIS SOFTWARE.



This license file must be retained with all copies of the software,

including any modified or derivative versions.
 */
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Emgu.CV;
using Emgu.CV.Structure;
using Emgu.Util;
using System.Drawing;
using System.Threading.Tasks;
using System.Collections.Concurrent;
namespace SiftLib
{
    public class Sift
    {

        const double SIFT_INIT_SIGMA = .5;
        const int SIFT_IMG_BORDER = 5;
        const int SIFT_MAX_INTERP_STEPS = 5;
        const int SIFT_ORI_HIST_BINS = 36;
        const double SIFT_ORI_RADIUS = 3 * SIFT_ORI_SIG_FCTR;
        const double SIFT_ORI_SIG_FCTR = 1.5;
        const int SIFT_ORI_SMOOTH_PASSES = 2;
        const double SIFT_ORI_PEAK_RATIO = .8;
        const double SIFT_DESCR_SCL_FCTR = 3.0;
        const double SIFT_DESCR_MAG_THR = .2;
        const double SIFT_INT_DESCR_FCTR = 512.0;


        const int SIFT_INTVLS = 3;
        const double SIFT_SIGMA = 1.6;
        const double SIFT_CONTR_THR = 0.04;
        const int SIFT_CURV_THR = 10;
        const int SIFT_IMG_DBL = 1;
        const int SIFT_DESCR_WIDTH = 4;
        const int SIFT_DESCR_HIST_BINS = 8;
        bool parallel;
        public List<Feature> sift_features(Image<Gray, float> img,bool para)
        {
            parallel = para;
            return _sift_features(img, SIFT_INTVLS, SIFT_SIGMA, SIFT_CONTR_THR,
                            SIFT_CURV_THR, SIFT_IMG_DBL, SIFT_DESCR_WIDTH,
                            SIFT_DESCR_HIST_BINS);
        }
        public List<Feature> sift_features(Image<Gray, float> img)
        {
            parallel = false;
            return _sift_features(img, SIFT_INTVLS, SIFT_SIGMA, SIFT_CONTR_THR,
                            SIFT_CURV_THR, SIFT_IMG_DBL, SIFT_DESCR_WIDTH,
                            SIFT_DESCR_HIST_BINS);
        }
        public List<Feature> _sift_features(Image<Gray, float> img, int intvls,
                       double sigma, double contr_thr, int curv_thr,
                       int img_dbl, int descr_width, int descr_hist_bins)
        {
            Image<Gray, Single> init_img;
            Image<Gray, Single>[,] gauss_pyr, dog_pyr;
            List<Feature> features;
            init_img = create_init_img(img, img_dbl, sigma);
            List<Feature> feat = new List<Feature>();
            int octvs = (int)(Math.Log(Math.Min(init_img.Width, init_img.Height)) / Math.Log(2) - 2);
            gauss_pyr = build_gauss_pyr(init_img, octvs, intvls, sigma);
            dog_pyr = build_dog_pyr(gauss_pyr, octvs, intvls);
            if(parallel)
            features = Pscale_space_extrema(dog_pyr, octvs, intvls, contr_thr, curv_thr);
            else
                features = Sscale_space_extrema(dog_pyr, octvs, intvls, contr_thr, curv_thr);
            calc_feature_scales(ref features, sigma, intvls);
            if (img_dbl != 0)
                adjust_for_img_dbl(ref features);

            calc_feature_oris(ref features,gauss_pyr);
            compute_descriptors(ref features, gauss_pyr, descr_width, descr_hist_bins);
            features.Sort(new Comparison<Feature>(delegate(Feature a, Feature b) { if (a.scl < b.scl)return 1; if (a.scl > b.scl) return -1; return 0; }));
            int n = features.Count;
            feat = features;


            return feat;

        }

       void compute_descriptors(ref List<Feature> features, Image<Gray, float>[,] gauss_pyr, int d, int n)
        {
            Feature feat;
            detection_data ddata;
            float[, ,] hist;
            int i, k = features.Count;

            for (i = 0; i < k; i++)
            {
                feat = features[i];
                ddata = feat.feature_data;
                hist = descr_hist(gauss_pyr[ddata.octv, ddata.intvl], ddata.r,
                    ddata.c, feat.ori, ddata.scl_octv, d, n);
                hist_to_descr(hist, d, n, ref feat);
                //release_descr_hist(&hist, d);
            }
        }
       void hist_to_descr(float[, ,] hist, int d, int n, ref Feature feat)
        {
            int int_val, i, r, c, o, k = 0;
            feat.descr = new double[d * d * n];

            for (r = 0; r < d; r++)
                for (c = 0; c < d; c++)
                    for (o = 0; o < n; o++)
                        feat.descr[k++] = hist[r, c, o];

            feat.d = k;
            normalize_descr(feat);
            for (i = 0; i < k; i++)
                if (feat.descr[i] > SIFT_DESCR_MAG_THR)
                    feat.descr[i] = SIFT_DESCR_MAG_THR;
            normalize_descr(feat);

            /* convert floating-point descriptor to integer valued descriptor */
            for (i = 0; i < k; i++)
            {
                int_val = (int)(SIFT_INT_DESCR_FCTR * feat.descr[i]);
                feat.descr[i] = Math.Min(255, int_val);
            }
        }
        void normalize_descr(Feature feat)
        {
            double cur, len_inv, len_sq = 0.0;
            int i, d = feat.d;

            for (i = 0; i < d; i++)
            {
                cur = feat.descr[i];
                len_sq += cur * cur;
            }
            len_inv = 1.0 / Math.Sqrt(len_sq);
            for (i = 0; i < d; i++)
                feat.descr[i] *= len_inv;
        }
        float[, ,] descr_hist(Image<Gray, float> img, int r, int c, double ori,
                     double scl, int d, int n)
        {
            float[, ,] hist;
            double cos_t, sin_t, hist_width, exp_denom, r_rot, c_rot, grad_mag,
                grad_ori, w, rbin, cbin, obin, bins_per_rad, PI2 = 2.0 * Math.PI;
            int radius, i, j;

            hist = new float[d, d, n];

            cos_t = Math.Cos(ori);
            sin_t = Math.Sin(ori);
            bins_per_rad = n / PI2;
            exp_denom = d * d * 0.5;
            hist_width = SIFT_DESCR_SCL_FCTR * scl;
            radius = (int)(hist_width * Math.Sqrt(2) * (d + 1.0) * 0.5 + 0.5);
            for (i = -radius; i <= radius; i++)
                for (j = -radius; j <= radius; j++)
                {
                    /*
                    Calculate sample's histogram array coords rotated relative to ori.
                    Subtract 0.5 so samples that fall e.g. in the center of row 1 (i.e.
                    r_rot = 1.5) have full weight placed in row 1 after interpolation.
                    */
                    c_rot = (j * cos_t - i * sin_t) / hist_width;
                    r_rot = (j * sin_t + i * cos_t) / hist_width;
                    rbin = r_rot + d / 2 - 0.5;
                    cbin = c_rot + d / 2 - 0.5;

                    if (rbin > -1.0 && rbin < d && cbin > -1.0 && cbin < d)
                        if (calc_grad_mag_ori(img, r + i, c + j, out grad_mag, out grad_ori) != 0)
                        {
                            grad_ori -= ori;
                            while (grad_ori < 0.0)
                                grad_ori += PI2;
                            while (grad_ori >= PI2)
                                grad_ori -= PI2;

                            obin = grad_ori * bins_per_rad;
                            w = Math.Exp(-(c_rot * c_rot + r_rot * r_rot) / exp_denom);
                            interp_hist_entry(ref hist, rbin, cbin, obin, grad_mag * w, d, n);
                        }
                }

            return hist;
        }
        void interp_hist_entry(ref float[, ,] hist, double rbin, double cbin,
                       double obin, double mag, int d, int n)
        {
            float d_r, d_c, d_o, v_r, v_c, v_o;

            int r0, c0, o0, rb, cb, ob, r, c, o;

            r0 = (int)Math.Floor(rbin);
            c0 = (int)Math.Floor(cbin);
            o0 = (int)Math.Floor(obin);
            d_r = (float)rbin - r0;
            d_c = (float)cbin - c0;
            d_o = (float)obin - o0;

            /*
            The entry is distributed into up to 8 bins.  Each entry into a bin
            is multiplied by a weight of 1 - d for each dimension, where d is the
            distance from the center value of the bin measured in bin units.
            */
            for (r = 0; r <= 1; r++)
            {
                rb = r0 + r;
                if (rb >= 0 && rb < d)
                {
                    v_r = (float)mag * ((r == 0) ? 1.0F - d_r : d_r);

                    for (c = 0; c <= 1; c++)
                    {
                        cb = c0 + c;
                        if (cb >= 0 && cb < d)
                        {
                            v_c = v_r * ((c == 0) ? 1.0F - d_c : d_c);

                            for (o = 0; o <= 1; o++)
                            {
                                ob = (o0 + o) % n;
                                v_o = v_c * ((o == 0) ? 1.0F - d_o : d_o);
                                hist[rb, cb, ob] += v_o;
                            }
                        }
                    }
                }
            }
        }

        void calc_feature_oris(ref List<Feature> features, Image<Gray, float>[,] gauss_pyr)
        {
            Feature feat;
            detection_data ddata;
            double[] hist;
            double omax;
            int i, j, n = features.Count;

            for (i = 0; i < n; i++)
            {
                feat = features[0];
                features.RemoveAt(0);
                ddata = feat.feature_data;
                hist = ori_hist(gauss_pyr[ddata.octv, ddata.intvl],
                                ddata.r, ddata.c, SIFT_ORI_HIST_BINS,
                                (int)Math.Round(SIFT_ORI_RADIUS * ddata.scl_octv),
                                SIFT_ORI_SIG_FCTR * ddata.scl_octv);
                for (j = 0; j < SIFT_ORI_SMOOTH_PASSES; j++)
                    smooth_ori_hist(ref hist, SIFT_ORI_HIST_BINS);
                omax = dominant_ori(ref  hist, SIFT_ORI_HIST_BINS);
                add_good_ori_features(ref features, hist, SIFT_ORI_HIST_BINS,
                                        omax * SIFT_ORI_PEAK_RATIO, feat);
                //free( ddata );
                //free( feat );
                //free( hist );
            }
        }
        void add_good_ori_features(ref List<Feature> features, double[] hist, int n,
                           double mag_thr, Feature feat)
        {
            Feature new_feat;
            double bin, PI2 = Math.PI * 2.0;
            int l, r, i;

            for (i = 0; i < n; i++)
            {
                l = (i == 0) ? n - 1 : i - 1;
                r = (i + 1) % n;

                if (hist[i] > hist[l] && hist[i] > hist[r] && hist[i] >= mag_thr)
                {
                    bin = i + interp_hist_peak(hist[l], hist[i], hist[r]);
                    bin = (bin < 0) ? n + bin : (bin >= n) ? bin - n : bin;
                    new_feat = (Feature)feat.Clone();
                    new_feat.ori = ((PI2 * bin) / n) - Math.PI;
                    features.Add(new_feat);
                    //free( new_feat );
                }
            }
        }
        double interp_hist_peak(double l, double c, double r) { return 0.5 * ((l) - (r)) / ((l) - 2.0 * (c) + (r)); }
        double dominant_ori(ref double[] hist, int n)
        {
            double omax;
            int maxbin, i;

            omax = hist[0];
            maxbin = 0;
            for (i = 1; i < n; i++)
                if (hist[i] > omax)
                {
                    omax = hist[i];
                    maxbin = i;
                }
            return omax;
        }
        void smooth_ori_hist(ref double[] hist, int n)
        {

            double prev, tmp, h0 = hist[0];
            int i;

            prev = hist[n - 1];
            for (i = 0; i < n; i++)
            {
                tmp = hist[i];
                hist[i] = 0.25 * prev + 0.5 * hist[i] +
                    0.25 * ((i + 1 == n) ? h0 : hist[i + 1]);
                prev = tmp;
            }
        }
        double[] ori_hist(Image<Gray, Single> img, int r, int c, int n, int rad, double sigma)
        {
            double[] hist;
            double mag, ori, w, exp_denom, PI2 = Math.PI * 2.0;
            int bin, i, j;

            hist = new double[n];
            exp_denom = 2.0 * sigma * sigma;
            for (i = -rad; i <= rad; i++)
                for (j = -rad; j <= rad; j++)
                    if (calc_grad_mag_ori(img, r + i, c + j, out mag, out ori) == 1)
                    {
                        w = Math.Exp(-(i * i + j * j) / exp_denom);
                        bin = (int)Math.Round(n * (ori + Math.PI) / PI2);
                        bin = (bin < n) ? bin : 0;
                        hist[bin] += w * mag;
                    }

            return hist;
        }
        int calc_grad_mag_ori(Image<Gray, Single> img, int r, int c, out double mag, out double ori)
        {
            double dx, dy;

            if (r > 0 && r < img.Height - 1 && c > 0 && c < img.Width - 1)
            {
                dx = img[r, c + 1].Intensity - img[r, c - 1].Intensity;
                dy = img[r - 1, c].Intensity - img[r + 1, c].Intensity;
                mag = Math.Sqrt(dx * dx + dy * dy);
                ori = Math.Atan2(dy, dx);
                return 1;
            }

            else
            {
                mag = 0;
                ori = 0;
                return 0;
            }
        }
        void adjust_for_img_dbl(ref List<Feature> features)
        {
            Feature feat;
            int i, n;

            n = features.Count;
            for (i = 0; i < n; i++)
            {
                feat = features[i];
                feat.x /= 2.0;
                feat.y /= 2.0;
                feat.scl /= 2.0;
                feat.img_pt.X /= 2.0F;
                feat.img_pt.Y /= 2.0F;
            }
        }
        void calc_feature_scales(ref List<Feature> features, double sigma, int intvls)
        {
            Feature feat;
            detection_data ddata;
            double intvl;
            int i, n;

            n = features.Count;
            for (i = 0; i < n; i++)
            {
                feat = features[i];
                intvl = feat.feature_data.intvl + feat.feature_data.subintvl;
                feat.scl = sigma * Math.Pow(2.0, feat.feature_data.octv + intvl / intvls);
                feat.feature_data.scl_octv = sigma * Math.Pow(2.0, intvl / intvls);
            }
        }


        Image<Gray, Single> create_init_img(Image<Gray, float> img, int img_dbl, double sigma)
        {
            Image<Gray, Single> gray;
            Image<Gray, Single> dbl;
            float sig_diff;

            gray = convert_to_gray32(img);
            if (img_dbl != 0)
            {
                sig_diff = (float)Math.Sqrt(sigma * sigma - SIFT_INIT_SIGMA * SIFT_INIT_SIGMA * 4);
                dbl = new Image<Gray, float>(new Size(img.Width * 2, img.Height * 2));
                dbl = gray.Resize(dbl.Width, dbl.Height, Emgu.CV.CvEnum.INTER.CV_INTER_CUBIC);
                dbl = dbl.SmoothGaussian(0, 0, sig_diff, sig_diff);
                return dbl;
            }
            else
            {
                sig_diff = (float)Math.Sqrt(sigma * sigma - SIFT_INIT_SIGMA * SIFT_INIT_SIGMA);
                gray.SmoothGaussian(0, 0, sig_diff, sig_diff);
                return gray;
            }
        }

        Image<Gray, Single> convert_to_gray32(Image<Gray, float> img)
        {
            Image<Gray, Byte> gray8;
            Image<Gray, Single> gray32;

            gray32 = new Image<Gray, Single>(img.Width, img.Height);

            using (gray8 = img.Convert<Gray, Byte>())
            {

                gray32 = gray8.ConvertScale<Single>(1.0 / 255.0, 0);
            }


            return gray32;
        }

        Image<Gray, Single>[,] build_gauss_pyr(Image<Gray, Single> basepic, int octvs,
                            int intvls, double sigma)
        {
            Image<Gray, Single>[,] gauss_pyr = new Image<Gray, float>[octvs, intvls + 3];
            double[] sig = new double[intvls + 3];
            double sig_total, sig_prev, k;
            int i, o;



            /*
                precompute Gaussian sigmas using the following formula:

                \sigma_{total}^2 = \sigma_{i}^2 + \sigma_{i-1}^2
            */
            sig[0] = sigma;
            k = Math.Pow(2.0, 1.0 / intvls);
            for (i = 1; i < intvls + 3; i++)
            {
                sig_prev = Math.Pow(k, i - 1) * sigma;
                sig_total = sig_prev * k;
                sig[i] = Math.Sqrt(sig_total * sig_total - sig_prev * sig_prev);
            }

            for (o = 0; o < octvs; o++)
                for (i = 0; i < intvls + 3; i++)
                {
                    if (o == 0 && i == 0)
                        gauss_pyr[o, i] = basepic.Clone();

                    /* base of new octvave is halved image from end of previous octave */
                    else if (i == 0)
                        gauss_pyr[o, i] = downsample(gauss_pyr[o - 1, intvls]);

                    /* blur the current octave's last image to create the next one */
                    else
                    {


                        gauss_pyr[o, i] = gauss_pyr[o, i - 1].SmoothGaussian(0, 0, sig[i], sig[i]);
                    }
                }


            return gauss_pyr;
        }
        Image<Gray, Single> downsample(Image<Gray, Single> img)
        {
            Image<Gray, Single> smaller;
            smaller = img.Resize(img.Width / 2, img.Height / 2, Emgu.CV.CvEnum.INTER.CV_INTER_NN);


            return smaller;
        }

       Image<Gray, Single>[,] build_dog_pyr(Image<Gray, Single>[,] gauss_pyr, int octvs, int intvls)
        {
            Image<Gray, Single>[,] dog_pyr;
            int i, o;

            dog_pyr = new Image<Gray, float>[octvs, intvls + 2];

            for (o = 0; o < octvs; o++)
                for (i = 0; i < intvls + 2; i++)
                {
                    dog_pyr[o, i] = gauss_pyr[o, i + 1].Sub(gauss_pyr[o, i]);
                }

            return dog_pyr;
        }
        List<Feature> Pscale_space_extrema(Image<Gray, Single>[,] dog_pyr, int octvs, int intvls,
                           double contr_thr, int curv_thr)
        {
            List<Feature> features = new List<Feature>();
            double prelim_contr_thr = 0.5 * contr_thr / intvls;
            object loc=new object();
            

            int o, i;

            for (o = 0; o < octvs; o++)
                for (i = 1; i <= intvls; i++)
                    Parallel.For(SIFT_IMG_BORDER, dog_pyr[o, 0].Height - SIFT_IMG_BORDER, delegate(int r)
                    {
                        Parallel.For(SIFT_IMG_BORDER, dog_pyr[o, 0].Width - SIFT_IMG_BORDER, delegate(int c)
                        /* perform preliminary check on contrast */
                        {
                            Feature feat;
                            detection_data ddata;
                            
                            if (Math.Abs(dog_pyr[o, i][r, c].Intensity) > prelim_contr_thr)
                            {
                                if (is_extremum(dog_pyr, o, i, r, c) == 1)
                                {
                                    feat = interp_extremum(dog_pyr, o, i, r, c, intvls, contr_thr);
                                    if (feat != null)
                                    {
                                        ddata = feat.feature_data;
                                        if ((is_too_edge_like(dog_pyr[ddata.octv, ddata.intvl],
                                            ddata.r, ddata.c, curv_thr) == 0))
                                        {
                                            if (feat != null)
                                            {
                                                lock (loc)
                                                {
                                                    
                                                    features.Insert(0, feat);//cvSeqPush( features, feat );
                                                }
                                            }
                                           


                                        }

                                    }
                                    //else
                                    //{
                                    //    Console.WriteLine("Beep");
                                    //}
                                }
                            }
                        });
                    });
            return features;
        }
        List<Feature> Sscale_space_extrema(Image<Gray, Single>[,] dog_pyr, int octvs, int intvls,
                           double contr_thr, int curv_thr)
        {
            List<Feature> features = new List<Feature>();
            double prelim_contr_thr = 0.5 * contr_thr / intvls;
            Feature feat;
            detection_data ddata;
            int o, i;


            for (o = 0; o < octvs; o++)
                for (i = 1; i <= intvls; i++)
                    for (int r = SIFT_IMG_BORDER; r < dog_pyr[o, 0].Height - SIFT_IMG_BORDER; r++)
                    {
                        for(int c=SIFT_IMG_BORDER;c< dog_pyr[o, 0].Width - SIFT_IMG_BORDER; c++)
                        /* perform preliminary check on contrast */
                        {
                            if (Math.Abs(dog_pyr[o, i][r, c].Intensity) > prelim_contr_thr)
                            {
                                if (is_extremum(dog_pyr, o, i, r, c) == 1)
                                {
                                    feat = interp_extremum(dog_pyr, o, i, r, c, intvls, contr_thr);
                                    if (feat != null)
                                    {
                                        ddata = feat.feature_data;
                                        if ((is_too_edge_like(dog_pyr[ddata.octv, ddata.intvl],
                                            ddata.r, ddata.c, curv_thr) == 0))
                                        {
                                            features.Insert(0, feat);//cvSeqPush( features, feat );
                                        }

                                    }
                                }
                            }
                        }
                    }
            return features;
        }
        int is_extremum(Image<Gray, Single>[,] dog_pyr, int octv, int intvl, int r, int c)
        {
            float val = (float)dog_pyr[octv, intvl][r, c].Intensity;
            int i, j, k;

            /* check for maximum */
            if (val > 0)
            {
                for (i = -1; i <= 1; i++)
                    for (j = -1; j <= 1; j++)
                        for (k = -1; k <= 1; k++)
                            if (val < dog_pyr[octv, intvl + i][r + j, c + k].Intensity)
                                return 0;
            }

            /* check for minimum */
            else
            {
                for (i = -1; i <= 1; i++)
                    for (j = -1; j <= 1; j++)
                        for (k = -1; k <= 1; k++)
                            if (val > dog_pyr[octv, intvl + i][r + j, c + k].Intensity)
                                return 0;
            }

            return 1;
        }
        Feature interp_extremum(Image<Gray, Single>[,] dog_pyr, int octv, int intvl,
                                int r, int c, int intvls, double contr_thr)
        {
            Feature feat;
            detection_data ddata;
            double xi = 0, xr = 0, xc = 0, contr;
            int i = 0;

            while (i < SIFT_MAX_INTERP_STEPS)
            {
                interp_step(dog_pyr, octv, intvl, r, c, out xi, out xr, out xc);
                if (Math.Abs(xi) < 0.5 && Math.Abs(xr) < 0.5 && Math.Abs(xc) < 0.5)
                    break;

                c += (int)Math.Round(xc);
                r += (int)Math.Round(xr);
                intvl += (int)Math.Round(xi);

                if (intvl < 1 ||
                    intvl > intvls ||
                    c < SIFT_IMG_BORDER ||
                    r < SIFT_IMG_BORDER ||
                    c >= dog_pyr[octv, 0].Width - SIFT_IMG_BORDER ||
                    r >= dog_pyr[octv, 0].Height - SIFT_IMG_BORDER)
                {
                    return null;
                }

                i++;
            }

            /* ensure convergence of interpolation */
            if (i >= SIFT_MAX_INTERP_STEPS)
                return null;

            contr = interp_contr(dog_pyr, octv, intvl, r, c, xi, xr, xc);
            if (Math.Abs(contr) < contr_thr / intvls)
                return null;

            feat = new_feature();
            ddata = feat.feature_data;
            feat.img_pt.X = (float)(feat.x = (c + xc) * Math.Pow(2.0, octv));
            feat.img_pt.Y = (float)(feat.y = (double)((r + xr) * Math.Pow(2.0, octv)));
            ddata.r = r;
            ddata.c = c;
            ddata.octv = octv;
            ddata.intvl = intvl;
            ddata.subintvl = xi;
            feat.feature_data = ddata;
            return feat;
        }
        Feature new_feature()
        {
            Feature feat = new Feature();
            detection_data ddata = new detection_data();


            feat.feature_data = ddata;
            feat.type = feature_type.FEATURE_LOWE;

            return feat;
        }
        void interp_step(Image<Gray, Single>[,] dog_pyr, int octv, int intvl, int r, int c,
                 out double xi, out double xr, out double xc)
        {
            Matrix<Double> dD, H, H_inv, X = new Matrix<double>(3, 1);
            double[] x = new double[] { 0, 0, 0 };

            dD = deriv_3D(dog_pyr, octv, intvl, r, c);
            H = hessian_3D(dog_pyr, octv, intvl, r, c);
            H_inv = H.Clone();


            CvInvoke.cvInvert(H, H_inv.Ptr, Emgu.CV.CvEnum.INVERT_METHOD.CV_SVD);
            unsafe
            {
                fixed (double* a = &x[0])
                {
                    CvInvoke.cvInitMatHeader(X.Ptr, 3, 1, Emgu.CV.CvEnum.MAT_DEPTH.CV_64F, new IntPtr(a), 0x7fffffff);
                }
            }
            CvInvoke.cvGEMM(H_inv.Ptr, dD.Ptr, -1, IntPtr.Zero, 0, X.Ptr, 0);

            //cvReleaseMat( &dD );
            //cvReleaseMat( &H );
            //cvReleaseMat( &H_inv );

            xi = x[2];
            xr = x[1];
            xc = x[0];
        }
        Matrix<Double> deriv_3D(Image<Gray, Single>[,] dog_pyr, int octv, int intvl, int r, int c)
        {
            Matrix<Double> dI;
            double dx, dy, ds;

            dx = (dog_pyr[octv, intvl][r, c + 1].Intensity -
                dog_pyr[octv, intvl][r, c - 1].Intensity) / 2.0;
            dy = (dog_pyr[octv, intvl][r + 1, c].Intensity -
                dog_pyr[octv, intvl][r - 1, c].Intensity) / 2.0;
            ds = (dog_pyr[octv, intvl + 1][r, c].Intensity -
                dog_pyr[octv, intvl - 1][r, c].Intensity) / 2.0;

            dI = new Matrix<Double>(3, 1);
            dI[0, 0] = dx;
            dI[1, 0] = dy;
            dI[2, 0] = ds;

            return dI;
        }
         Matrix<Double> hessian_3D(Image<Gray, Single>[,] dog_pyr, int octv, int intvl, int r, int c)
        {
            Matrix<Double> H;
            double v, dxx, dyy, dss, dxy, dxs, dys;

            v = dog_pyr[octv, intvl][r, c].Intensity;
            dxx = dog_pyr[octv, intvl][r, c + 1].Intensity + dog_pyr[octv, intvl][r, c - 1].Intensity - 2 * v;
            dyy = dog_pyr[octv, intvl][r + 1, c].Intensity +
                    dog_pyr[octv, intvl][r - 1, c].Intensity - 2 * v;
            dss = dog_pyr[octv, intvl + 1][r, c].Intensity +
                    dog_pyr[octv, intvl - 1][r, c].Intensity - 2 * v;
            dxy = (dog_pyr[octv, intvl][r + 1, c + 1].Intensity -
                    dog_pyr[octv, intvl][r + 1, c - 1].Intensity -
                    dog_pyr[octv, intvl][r - 1, c + 1].Intensity +
                    dog_pyr[octv, intvl][r - 1, c - 1].Intensity) / 4.0;
            dxs = (dog_pyr[octv, intvl + 1][r, c + 1].Intensity -
                    dog_pyr[octv, intvl + 1][r, c - 1].Intensity -
                    dog_pyr[octv, intvl - 1][r, c + 1].Intensity +
                    dog_pyr[octv, intvl - 1][r, c - 1].Intensity) / 4.0;
            dys = (dog_pyr[octv, intvl + 1][r + 1, c].Intensity -
                    dog_pyr[octv, intvl + 1][r - 1, c].Intensity -
                    dog_pyr[octv, intvl - 1][r + 1, c].Intensity +
                    dog_pyr[octv, intvl - 1][r - 1, c].Intensity) / 4.0;

            H = new Matrix<double>(3, 3);
            H[0, 0] = dxx;
            H[0, 1] = dxy;
            H[0, 2] = dxs;
            H[1, 0] = dxy;
            H[1, 1] = dyy;
            H[1, 2] = dys;
            H[2, 0] = dxs;
            H[2, 1] = dys;
            H[2, 2] = dss;

            return H;
        }
        double interp_contr(Image<Gray, Single>[,] dog_pyr, int octv, int intvl, int r,
                    int c, double xi, double xr, double xc)
        {
            Matrix<double> dD, X = new Matrix<double>(3, 1), T = new Matrix<double>(1, 1);
            double[] t = new double[1];
            double[] x = new double[3] { xc, xr, xi };

            unsafe
            {
                fixed (double* a = &x[0])
                {
                    CvInvoke.cvInitMatHeader(X.Ptr, 3, 1, Emgu.CV.CvEnum.MAT_DEPTH.CV_64F, new IntPtr(a), 0x7fffffff);
                }
            }
            unsafe
            {
                fixed (double* a = &t[0])
                {
                    CvInvoke.cvInitMatHeader(T.Ptr, 1, 1, Emgu.CV.CvEnum.MAT_DEPTH.CV_64F, new IntPtr(a), 0x7fffffff);
                }
            }
            dD = deriv_3D(dog_pyr, octv, intvl, r, c);
            CvInvoke.cvGEMM(dD.Ptr, X.Ptr, 1, IntPtr.Zero, 0, T.Ptr, Emgu.CV.CvEnum.GEMM_TYPE.CV_GEMM_A_T);
            //cvReleaseMat( &dD );

            return dog_pyr[octv, intvl][r, c].Intensity + t[0] * 0.5;
        }
        int is_too_edge_like(Image<Gray, float> dog_img, int r, int c, int curv_thr)
        {
            double d, dxx, dyy, dxy, tr, det;
            /*
             * BT ADDED
             *
             * */
            if ((c == 0) || (r == 0))
                return 1;
            /*
             * BT ENDED
             * /
            /* principal curvatures are computed using the trace and det of Hessian */
            d = dog_img[r, c].Intensity;
            dxx = dog_img[r, c + 1].Intensity + dog_img[r, c - 1].Intensity - 2 * d;
            dyy = dog_img[r + 1, c].Intensity + dog_img[r - 1, c].Intensity - 2 * d;
            dxy = (dog_img[r + 1, c + 1].Intensity - dog_img[r + 1, c - 1].Intensity -
                    dog_img[r - 1, c + 1].Intensity + dog_img[r - 1, c - 1].Intensity) / 4.0;
            tr = dxx + dyy;
            det = dxx * dyy - dxy * dxy;

            /* negative determinant -> curvatures have different signs; reject Feature */
            if (det <= 0)
                return 1;

            if (tr * tr / det < (curv_thr + 1.0) * (curv_thr + 1.0) / curv_thr)
                return 0;
            return 1;
        }

    }
}
