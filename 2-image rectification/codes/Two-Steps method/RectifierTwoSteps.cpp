#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <Eigen/Dense>
#include <Eigen/SVD>

 
using namespace cv;
using namespace std;
using namespace Eigen;


struct Pointi {
	float x, y;
};

struct colori {
    float blue;
    float green;
    float red;
};



tuple<float, float> scale_image(float width, float height, float fix_dimension);
bool point_in_polygon(Vector3f point, vector<Pointi> polygon);
colori color_estimator(float x_coor, float y_coor, Mat image);



int main() {
    
    // For scaling output image size
    float fix_dimension = 2000;
    
    string AFFINE_RESULT_PATH = "../../images/results/result-2stepAffined.jpg";
    string RESULT_PATH = "../../images/results/result-2stepSimilar.jpg";
    
    // Read transfered image (prime coordinates)
    string projected_path = "../../images/Img2.jpeg";
    Mat projected_img = imread(projected_path, IMREAD_COLOR);
    
    // Remove projective transform
    // Four points to find vanishing line       
//    Vector3f P_pt(318, 230, 1);
//    Vector3f Q_pt(212, 860, 1);
//    Vector3f R_pt(875, 1130, 1);
//    Vector3f S_pt(1044, 230, 1);
    
    Vector3f P_pt(368, 552, 1);
    Vector3f Q_pt(363, 852, 1);
    Vector3f R_pt(642, 973, 1);
    Vector3f S_pt(621, 509, 1);
    
    // VL Line
    Vector3f PQ_ln = (P_pt.cross(Q_pt)).normalized();
    Vector3f SR_ln = (S_pt.cross(R_pt)).normalized();
    Vector3f van_pt1 = PQ_ln.cross(SR_ln);
    
    // Lines
    Vector3f PS_ln = (P_pt.cross(S_pt)).normalized();
    Vector3f QR_ln = (Q_pt.cross(R_pt)).normalized();
    Vector3f van_pt2 = PS_ln.cross(QR_ln);
    
    // Vanishing line
    Vector3f van_line = (van_pt1.cross(van_pt2)).normalized();
    
    // H matrix
    Matrix<float, 3, 3> H_p {{1, 0, 0},
                             {0, 1, 1},
                             {van_line(0,0), van_line(1,0), van_line(2,0)}};

//    cout << H_p << endl;
    
    // Create a scaled blank image for rectified image
    vector<Pointi> projected_img_corners = {{0,0},
                                            {0, (float)projected_img.rows},
                                            {(float)projected_img.cols, (float)projected_img.rows},
                                            {(float)projected_img.cols,0}};    
    
    float x_list_p[4];
    float y_list_p[4];
    for (int i=0; i<4; i++) {
//        cout << projected_img_corners(i,0) << "---" << projected_img_corners(i,1) << endl;
        
        // Find corresponding point by using homography matrix
        Vector3f homog_point(projected_img_corners[i].x, projected_img_corners[i].y, 1);
        Vector3f point_p = H_p * homog_point;
        point_p /= point_p(2);
        
        x_list_p[i] = point_p(0);
        y_list_p[i] = point_p(1);
    }
    
    float max_x_p = *max_element(x_list_p, x_list_p+4);
    float min_x_p = *min_element(x_list_p, x_list_p+4);
    float max_y_p = *max_element(y_list_p, y_list_p+4);
    float min_y_p = *min_element(y_list_p, y_list_p+4);    
   
    float width_p  = max_x_p - min_x_p;
    float height_p = max_y_p - min_y_p;    
//    cout <<  width_p << ' ' << height_p << endl;
   
    auto [scaled_width_p, scaled_height_p] = scale_image(width_p,
                                                     height_p,
                                                     fix_dimension);
//    cout << scaled_width << ' ' << scaled_height << endl;
    
    // The amount of size changing
    float w_change_p = width_p  / scaled_width_p;
    float h_change_p = height_p / scaled_height_p;
    
    // Rectified image
    Mat affined_img((int)scaled_height_p, (int)scaled_width_p, CV_8UC3, Scalar(0, 0, 0));
    
    
    // Find color on source image by checking if the point is in ROI polygone    
    for (int i=0; i<affined_img.cols; i++) {
        
        for (int j=0; j<affined_img.rows; j++) {
            
            float x_p = (i * w_change_p) + min_x_p;
            float y_p = (j * h_change_p) + min_y_p;
            
            // Find corresponding point by using homography matrix
            Vector3f prime_point(x_p, y_p, 1);
            Vector3f point_before = H_p.inverse() * prime_point;
            point_before /= point_before(2);
                                                                            
//            cout << point_before.x << ' ' << point_before.y << endl;      
                                                                
            if (point_in_polygon(point_before, projected_img_corners)) {
//                cout << point_before.x << ' ' << point_before.y << endl;
                
                if (point_before(0) >= 0 & point_before(1) >= 0) {
                
                    colori est_color = color_estimator(point_before(0), point_before(1), projected_img);
                   
                    // Make rectified image
                    affined_img.at<Vec3b>( j , i )[0] = est_color.blue;
                    affined_img.at<Vec3b>( j , i )[1] = est_color.green;
                    affined_img.at<Vec3b>( j , i )[2] = est_color.red;
                    
                 }
                
                
            } else {
                continue;
            }	        
            
        }
    }
    
    // Save result image
    imwrite(AFFINE_RESULT_PATH, affined_img);
    cout << "\nProjective Transform Removed!\n" << endl;
    
    
    
    
    
    // Remove affine transform
    // Four points to find orthogonal lines      
    
    Vector3f P2_pt = H_p * P_pt;
//    P2_pt /= P2_pt(2);
    Vector3f Q2_pt = H_p * Q_pt;
//    Q2_pt /= Q2_pt(2);
    Vector3f R2_pt = H_p * R_pt;
//    R2_pt /= R2_pt(2);
    Vector3f S2_pt = H_p * S_pt;
//    S2_pt /= S2_pt(2);
    
    Vector3f l_prime1 = (P2_pt.cross(Q2_pt)).normalized();
    Vector3f m_prime1 = (Q2_pt.cross(R2_pt)).normalized();
    Vector3f l_prime2 = (R2_pt.cross(S2_pt)).normalized();
    Vector3f m_prime2 = (S2_pt.cross(P2_pt)).normalized();
                                    
    // MS = B 
    float m11 = l_prime1(0) * m_prime1(0);
    float m12 = l_prime1(0) * m_prime1(1) + l_prime1(1) * m_prime1(0);
    float b1 = -l_prime1(1) * m_prime1(1);
    
    float m21 = l_prime2(0) * m_prime2(0);
    float m22 = l_prime2(0) * m_prime2(1) + l_prime2(1) * m_prime2(0);
    float b2 = -l_prime2(1) * m_prime2(1);
                                    
    Matrix<float, 2, 2> M {{m11, m12},
                           {m21, m22}};

    Vector2f B(b1, b2);
    Vector2f S = M.inverse() * B;
    
    // AA.t() matrix
    Matrix<float, 2, 2> AAt {{S(0) , S(1)},
                             {S(1) , 1}};
    
    // Compute the SVD of AAt
    JacobiSVD<Matrix2f> svd(AAt, ComputeFullU | ComputeFullV);
//    Matrix2f U = svd.matrixU();
    Vector2f W = svd.singularValues();
    Matrix2f V = svd.matrixV();
    
    // D matrix
    Matrix<float, 2, 2> D {{sqrt(W(0)), 0},
                           {0, sqrt(W(1))}};
    // A matrix
    Matrix2f A = V * D * V.transpose();
    
    // H matrix    
    Matrix<float, 3, 3> H_affine {{A(0,0), A(0,1), 0},
                                  {A(1,0), A(1,1), 0},
                                  {0, 0, 1}};
                            
    // Totally H matrix to "Remove" projective and affine transform    
    Matrix3f H = H_affine.inverse() * H_p;
    
    // Find corner points on image and map them to world plane
    float x_list_af[4];
    float y_list_af[4];
    for (int i=0; i<4; i++) {
        
        // Find corresponding point by using homography matrix                                               
        Vector3f homog_point(projected_img_corners[i].x, projected_img_corners[i].y, 1);
        Vector3f point_orig = H * homog_point;
        point_orig /= point_orig(2);
        
        x_list_af[i] = point_orig(0);
        y_list_af[i] = point_orig(1);
        
    }
    
    float max_x_orig = *max_element(x_list_af, x_list_af+4);
    float min_x_orig = *min_element(x_list_af, x_list_af+4);
    float max_y_orig = *max_element(y_list_af, y_list_af+4);
    float min_y_orig = *min_element(y_list_af, y_list_af+4);
    
    float width_orig  = max_x_orig - min_x_orig;
    float height_orig = max_y_orig - min_y_orig;    
//    cout << width_orig << ' ' << height_orig << endl;
    
    // Create a scaled blank image for rectified image   
    auto [scaled_width, scaled_height] = scale_image(width_orig,
                                                      height_orig,
                                                      fix_dimension);
//    cout << scaled_width << ' ' << scaled_height << endl;
    
    // The amount of size changing
    float w_change = width_orig  / scaled_width;
    float h_change = height_orig / scaled_height;
    
    Mat orig_img((int)scaled_height, (int)scaled_width, CV_8UC3, Scalar(0, 0, 0));
    
    
    // Find color on source image by checking if the point is in ROI polygone    
    for (int i=0; i<orig_img.cols; i++) {
        
        for (int j=0; j<orig_img.rows; j++) {

            float x_orig = (i * w_change) + min_x_orig;
            float y_orig = (j * h_change) + min_y_orig;            
            
            // Find corresponding point by using homography matrix                      
            Vector3f orig_point(x_orig, y_orig, 1);
            Vector3f prime_point = H.inverse() * orig_point;
            prime_point /= prime_point(2);
            
//            cout << prime_point(0) << ' ' << prime_point(1) << endl;                                           
                                                                 
                                                                
            if (point_in_polygon(prime_point, projected_img_corners)) {
                

            
                if (prime_point(0) >= 0 & prime_point(1) >= 0) {
                    
                    colori est_color = color_estimator(prime_point(0),
                                                       prime_point(1),
                                                       projected_img);
                                                       
//                    cout << est_color.blue << ' ' << est_color.green << ' ' << est_color.red << endl;
                        
                    // Make rectified image
                    orig_img.at<Vec3b>( j,  i)[0] = est_color.blue;
                    orig_img.at<Vec3b>( j,  i)[1] = est_color.green;
                    orig_img.at<Vec3b>( j,  i)[2] = est_color.red;                    
                    
                 }
                
            } else {
                continue;
            }
	        
            
        }
    }
    
    
    // Save result image
    imwrite(RESULT_PATH, orig_img);
    cout << "Affine Transform Removed!\n" << endl;



    return 0;
}




tuple<float, float> scale_image(float width, float height, float fix_dimension) {

    float scaled_width;
    float scaled_height;
    if ((height > width) & (height >= fix_dimension)) {
        
        scaled_height = fix_dimension;
        float aspect_ratio = width / height;
        scaled_width = scaled_height * aspect_ratio;
    
    } else if ((width > height) & (width >= fix_dimension)) {
    
        scaled_width = fix_dimension;
        float aspect_ratio = height / width;
        scaled_height = scaled_width * aspect_ratio;
            
    } else {
    
        if (height > width) {
            scaled_height = fix_dimension;
            float aspect_ratio = width / height;
            scaled_width = scaled_height * aspect_ratio;
        } else {
            scaled_width = fix_dimension;
            float aspect_ratio = height / width;
            scaled_height = scaled_width * aspect_ratio;
        }
    
    }
    
    return {scaled_width, scaled_height};

}



// Checking if a point is inside a polygon
bool point_in_polygon(Vector3f point, vector<Pointi> polygon){

	int num_vertices = polygon.size();
	float x = point(0), y = point(1);
	bool inside = false;

	// Store the first point in the polygon and initialize
	// the second point
	Pointi p1 = polygon[0], p2;

	// Loop through each edge in the polygon
	for (int i = 1; i <= num_vertices; i++) {
		// Get the next point in the polygon
		p2 = polygon[i % num_vertices];
		
//		if (p2.x == x && p2.y == y) {
//		    inside = true;
//		    break;
//		}
		// Check if the point is above the minimum y
		// coordinate of the edge
		if (y > min(p1.y, p2.y)) {
			// Check if the point is below the maximum y
			// coordinate of the edge
			if (y <= max(p1.y, p2.y)) {
				// Check if the point is to the left of the
				// maximum x coordinate of the edge
				if (x <= max(p1.x, p2.x)) {
					// Calculate the x-intersection of the
					// line connecting the point to the edge
					float x_intersection
						= (y - p1.y) * (p2.x - p1.x)
							/ (p2.y - p1.y)
						+ p1.x;

					// Check if the point is on the same
					// line as the edge or to the left of
					// the x-intersection
					if (p1.x == p2.x
						|| x <= x_intersection) {
						// Flip the inside flag
						inside = !inside;
					}
				}
			}
		}

		// Store the current point as the first point for
		// the next iteration
		p1 = p2;
	}

	// Return the value of the inside flag
	return inside;
}



// Calculate the distance between two points
float distance(float point1[], float point2[]) {
    
    return sqrt( pow(point1[0]-point2[0], 2) + pow(point1[1]-point2[1], 2) );   
}



// Estimate the color by bilinear interpolation method
colori color_estimator(float x_coor, float y_coor, Mat image) {
    
    float point_coor[2] = {y_coor, x_coor};
    colori estimate_color;

    float BI_point1[2] = {floor(y_coor) , floor(x_coor)};
    Vec3b BIP1_intensity = image.at<Vec3b>(BI_point1[0], BI_point1[1]);
    float BIP1_blue = BIP1_intensity.val[0];
    float BIP1_green = BIP1_intensity.val[1];
    float BIP1_red = BIP1_intensity.val[2];
    float BIP1_dist = distance(BI_point1, point_coor);
//    float BIP1_dist = sqrt( (BI_point1[0]-y_coor)**2 + (BI_point1[1]-x_coor)**2 );
//        cout << "orig colors = " << red << ' ' << green << ' ' << blue << ' ' << endl;
    
    float BI_point2[2] = {floor(y_coor) , ceil(x_coor)};
    Vec3b BIP2_intensity = image.at<Vec3b>(BI_point2[0], BI_point2[1]);
    float BIP2_blue = BIP2_intensity.val[0];
    float BIP2_green = BIP2_intensity.val[1];
    float BIP2_red = BIP2_intensity.val[2];
    float BIP2_dist = distance(BI_point2, point_coor);
    
    float BI_point3[2] = {ceil(y_coor) , floor(x_coor)};
    Vec3b BIP3_intensity = image.at<Vec3b>(BI_point3[0], BI_point3[1]);
    float BIP3_blue = BIP3_intensity.val[0];
    float BIP3_green = BIP3_intensity.val[1];
    float BIP3_red = BIP3_intensity.val[2];
    float BIP3_dist = distance(BI_point3, point_coor);;
    
    float BI_point4[2] = {ceil(y_coor) , ceil(x_coor)};
    Vec3b BIP4_intensity = image.at<Vec3b>(BI_point4[0], BI_point4[1]);
    float BIP4_blue = BIP4_intensity.val[0];
    float BIP4_green = BIP4_intensity.val[1];
    float BIP4_red = BIP4_intensity.val[2];
    float BIP4_dist = distance(BI_point4, point_coor);
    
    float denom = 1/BIP1_dist + 1/BIP2_dist + 1/BIP3_dist + 1/BIP4_dist;
    float estimate_blue = (BIP1_blue/BIP1_dist + BIP2_blue/BIP2_dist + BIP3_blue/BIP3_dist + BIP4_blue/BIP4_dist) / denom;
    float estimate_green = (BIP1_green/BIP1_dist + BIP2_green/BIP2_dist + BIP3_green/BIP3_dist + BIP4_green/BIP4_dist) / denom;
    float estimate_red = (BIP1_red/BIP1_dist + BIP2_red/BIP2_dist + BIP3_red/BIP3_dist + BIP4_red/BIP4_dist) / denom;
    
    estimate_color.blue = estimate_blue;
    estimate_color.green = estimate_green;
    estimate_color.red = estimate_red;
    
    return estimate_color;

}














