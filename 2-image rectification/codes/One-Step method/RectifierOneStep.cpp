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
    
    string RESULT_PATH = "../../images/results/result-1step.jpg";
    
    // Read transfered image (prime coordinates)
    string projected_path = "../../images/Img2.jpeg";
    Mat projected_img = imread(projected_path, IMREAD_COLOR);
    
    // Estimate output image size
    vector<Pointi> corner_points = {{0,0},
                                    {0, (float)projected_img.rows},
                                    {(float)projected_img.cols, (float)projected_img.rows},
                                    {(float)projected_img.cols, 0}};

    // Remove homographic transform
    Vector3f P_pt(382, 575, 1);
    Vector3f Q_pt(379,835, 1);
    Vector3f R_pt(607,923, 1);
    Vector3f S_pt(594,551, 1);
    Vector3f T_pt(425,231, 1);
    Vector3f U_pt(425,347, 1);
    Vector3f V_pt(515,298, 1);

    // Two pairs orthogonal lines (PQ-QR, RS-SP, TU-UV, QR-RS, SP-PQ)
    Vector3f l_prime1 = (P_pt.cross(Q_pt)).normalized();
    Vector3f m_prime1 = (Q_pt.cross(R_pt)).normalized(); //////////////
    
    Vector3f l_prime2 = (R_pt.cross(S_pt)).normalized();
    Vector3f m_prime2 = (S_pt.cross(P_pt)).normalized();
    
    Vector3f l_prime3 = (T_pt.cross(U_pt)).normalized();
    Vector3f m_prime3 = (U_pt.cross(V_pt)).normalized();
    
    Vector3f l_prime4 = m_prime1;
    Vector3f m_prime4 = l_prime2;
    
    Vector3f l_prime5 = m_prime2;
    Vector3f m_prime5 = l_prime1;

    
    // building
//    Vector3f P_pt(237, 196, 1);
//    Vector3f Q_pt(232, 370, 1);
//    Vector3f R_pt(295, 375, 1);
//    Vector3f S_pt(301, 213, 1);
//    Vector3f T_pt(232,378, 1);
//    Vector3f U_pt(240,124, 1);
//    Vector3f V_pt(716,291, 1);  
//    
//    // Orthogonal lines
//    Vector3f l_prime1 = (P_pt.cross(Q_pt)).normalized();
//    Vector3f m_prime1 = (Q_pt.cross(R_pt)).normalized(); //////////////
//    
//    Vector3f l_prime2 = (R_pt.cross(S_pt)).normalized();
//    Vector3f m_prime2 = (S_pt.cross(P_pt)).normalized();
//    
//    Vector3f l_prime3 = (T_pt.cross(U_pt)).normalized();
//    Vector3f m_prime3 = (U_pt.cross(V_pt)).normalized();
//    
//    Vector3f l_prime4 = m_prime1;
//    Vector3f m_prime4 = l_prime2;
//    
//    Vector3f l_prime5 = m_prime2;
//    Vector3f m_prime5 = l_prime1;
    
    
    
    
                                    
    
    
    // MS = B
    float m11 = l_prime1(0) * m_prime1(0);
    float m12 = (l_prime1(0) * m_prime1(1) + l_prime1(1) * m_prime1(0));
    float m13 = l_prime1(1) * m_prime1(1);
    float m14 = (l_prime1(0) * m_prime1(2) + l_prime1(2) * m_prime1(0));
    float m15 = (l_prime1(1) * m_prime1(2) + l_prime1(2) * m_prime1(1));
    float b1  = -l_prime1(2) * m_prime1(2);
    
    float m21 = l_prime2(0) * m_prime2(0);
    float m22 = (l_prime2(0) * m_prime2(1) + l_prime2(1) * m_prime2(0));
    float m23 = l_prime2(1) * m_prime2(1);
    float m24 = (l_prime2(0) * m_prime2(2) + l_prime2(2) * m_prime2(0));
    float m25 = (l_prime2(1) * m_prime2(2) + l_prime2(2) * m_prime2(1));
    float b2  = -l_prime2(2) * m_prime2(2);
    
    float m31 = l_prime3(0) * m_prime3(0);
    float m32 = (l_prime3(0) * m_prime3(1) + l_prime3(1) * m_prime3(0));
    float m33 = l_prime3(1) * m_prime3(1);
    float m34 = (l_prime3(0) * m_prime3(2) + l_prime3(2) * m_prime3(0));
    float m35 = (l_prime3(1) * m_prime3(2) + l_prime3(2) * m_prime3(1));
    float b3  = -l_prime3(2) * m_prime3(2);
    
    float m41 = l_prime4(0) * m_prime4(0);
    float m42 = (l_prime4(0) * m_prime4(1) + l_prime4(1) * m_prime4(0));
    float m43 = l_prime4(1) * m_prime4(1);
    float m44 = (l_prime4(0) * m_prime4(2) + l_prime4(2) * m_prime4(0));
    float m45 = (l_prime4(1) * m_prime4(2) + l_prime4(2) * m_prime4(1));
    float b4  = -l_prime4(2) * m_prime4(2);
    
    float m51 = l_prime5(0) * m_prime5(0);
    float m52 = (l_prime5(0) * m_prime5(1) + l_prime5(1) * m_prime5(0));
    float m53 = l_prime5(1) * m_prime5(1);
    float m54 = (l_prime5(0) * m_prime5(2) + l_prime5(2) * m_prime5(0));
    float m55 = (l_prime5(1) * m_prime5(2) + l_prime5(2) * m_prime5(1));
    float b5  = -l_prime5(2) * m_prime5(2);

    
    Matrix<float, 5, 5> M {{m11 , m12, m13, m14, m15},
                           {m21 , m22, m23, m24, m25},
                           {m31 , m32, m33, m34, m35},
                           {m41 , m42, m43, m44, m45},
                           {m51 , m52, m53, m54, m55}};
    
    VectorXf B(5); B << b1, b2, b3, b4, b5;
    VectorXf S = (M.inverse() * B).normalized(); ///////////////
//    cout << S << endl;
    

    
    // AA.t() matrix    
    Matrix<float, 2, 2> AAt {{S(0,0) , S(1,0)/2},
                             {S(1,0)/2 , S(2,0)}}; //////////////
                             
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
    
    // v matrix
    Vector2f vv(S(3,0)/2, S(4,0)/2);
    Vector2f v = A.inverse() * vv;
    
    // H matrix
    Matrix<float, 3, 3> H {{A(0,0), A(0,1), 0},
                           {A(1,0), A(1,1), 0},
                           {v(0,0), v(1,0), 1}};
                           
//    cout << H << endl;
    
    // Find corner points on image and map them to world plane
    float x_list[4];
    float y_list[4];
    for (int i=0; i<4; i++) {
        
        // Find corresponding point by using homography matrix                                               
        Vector3f homog_point(corner_points[i].x, corner_points[i].y, 1);
        Vector3f point_orig = H.inverse() * homog_point;
        point_orig /= point_orig(2);
        
        x_list[i] = point_orig(0);
        y_list[i] = point_orig(1);
        
    }
    
    float max_x = *max_element(x_list, x_list+4);
    float min_x = *min_element(x_list, x_list+4);
    float max_y = *max_element(y_list, y_list+4);
    float min_y = *min_element(y_list, y_list+4);
    
    float width  = max_x - min_x;
    float height = max_y - min_y;    
    cout << width << ' ' << height << endl;

    // Create a scaled blank image for rectified image   
    auto [scaled_width, scaled_height] = scale_image(width,
                                                      height,
                                                      fix_dimension);
//    cout << scaled_width << ' ' << scaled_height << endl;
    
    // The amount of size changing
    float w_change = width  / scaled_width;
    float h_change = height / scaled_height;
    
    Mat orig_img((int)scaled_height, (int)scaled_width, CV_8UC3, Scalar(0, 0, 0));
    
    // Find color on source image by checking if the point is in ROI polygone    
    for (int i=0; i<orig_img.cols; i++) {
        
        for (int j=0; j<orig_img.rows; j++) {

            float x_orig = (i * w_change) + min_x;
            float y_orig = (j * h_change) + min_y;            
            
            // Find corresponding point by using homography matrix                      
            Vector3f orig_point(x_orig, y_orig, 1);
            Vector3f prime_point = H * orig_point;
            prime_point /= prime_point(2);
            
//            cout << prime_point(0) << ' ' << prime_point(1) << endl;                                           
                                                                 
                                                                
            if (point_in_polygon(prime_point, corner_points)) {
                

            
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


