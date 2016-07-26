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
using System.Drawing;
using Emgu.CV.Structure;
using Emgu.CV;
using Emgu.Util;
namespace SiftLib
{
    public class Feature : ICloneable
    {
        public const int FEATURE_MAX_D = 128;
        public double x;                      /**< x coord */
        public double y;                      /**< y coord */
        public double a;                      /**< Oxford-type affine region parameter */
        public double b;                      /**< Oxford-type affine region parameter */
        public double c;                      /**< Oxford-type affine region parameter */
        public double scl;                    /**< scale of a Lowe-style feature */
        public double ori;                    /**< orientation of a Lowe-style feature */
        public int d;                         /**< descriptor length */
        public double[] descr;   /**< descriptor */
        public feature_type type;                      /**< feature type, OXFD or LOWE */
        public int category;                  /**< all-purpose feature category */
        public Feature fwd_match;     /**< matching feature from forward image */
        public Feature bck_match;     /**< matching feature from backmward image */
        public Feature mdl_match;     /**< matching feature from model */
        public PointF img_pt;           /**< location in image */
        public MCvPoint2D64f mdl_pt;           /**< location in model */
        public detection_data feature_data;            /**< user-definable data */
        public Feature()
        {
            descr = new double[FEATURE_MAX_D];
        }
        
        #region ICloneable Members

        public object Clone()
        {
            Feature feat = new Feature();
            feat.a = a;
            feat.b = b;
            feat.bck_match = bck_match;
            feat.c = c;
            feat.category = category;
            feat.d = d;
            feat.descr = (double[]) descr.Clone();
            feat.feature_data = (detection_data)feature_data.Clone();
            feat.fwd_match = fwd_match;
            feat.img_pt = img_pt;
            feat.mdl_match = mdl_match;
            feat.mdl_pt = mdl_pt;
            feat.ori = ori;
            feat.scl = scl;
            feat.type = type;
            feat.x = x;
            feat.y = y;
            return feat;
        }

        #endregion
    }
    public enum feature_type
    {
        FEATURE_OXFD,
        FEATURE_LOWE,
    };
    public struct detection_data : ICloneable
    {
        public int r;
        public int c;
        public int octv;
        public int intvl;
        public double subintvl;
        public double scl_octv;

        #region ICloneable Members

        public object Clone()
        {
            detection_data dat = new detection_data();
            dat.c = c;
            dat.intvl = intvl;
            dat.octv = octv;
            dat.r = r;
            dat.scl_octv = scl_octv;
            dat.subintvl = subintvl;
            return dat;
        }

        #endregion
    }
}
