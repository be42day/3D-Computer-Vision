#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

 
using namespace cv;
using namespace std;



struct Pointi {
	float x, y;
};

struct colori {
    float blue;
    float green;
    float red;
};



Pointi calc_corresponding_point(float x, float y, float z, Mat H);
bool point_in_polygon(Pointi point, vector<Pointi> polygon);
float distance(float point1[], float point2[]);
colori color_estimator(float x_coor, float y_coor, Mat image);


int main() {
    // To scale so large output images (fixing the output image height)
    int scaled_height = 2000;
    
    string RESULT_PATH = "../../images/results/Rectified-H-inverse.jpg";
    
    // Read transfered image (prime coordinates)
    string transfered_path = "../../images/Img1.JPG";
    Mat projected_img = imread(transfered_path, IMREAD_COLOR);
           
    // Source points on projected image                            
    int src_points[4][2] = {{47, 307},
                            {8, 575},
                            {1011, 698},
                            {1000, 523}};
                           
    // Corresponding points on rectified image (real measurments)
    vector<Pointi> rec_corr_points = {{ 0 , 0 },
                                      { 0 , 85 },
                                      { 75 , 85 },
                                      { 75 , 0 }};
                                
    // Points for estimating output image size
    vector<Pointi> corner_points = {{ 0 , 0 },
                                    { 0 , (float)projected_img.rows },
                                    { (float)projected_img.cols , (float)projected_img.rows },
                                    { (float)projected_img.cols , 0 }};
                                
    // Homography matrix
    // To solve Ah = b equation
    
    int x_pr, y_pr;       
    Mat A;
    Mat b;
    for (int ii=0; ii<4; ii++){
        x_pr = rec_corr_points[ii].x;
        y_pr = rec_corr_points[ii].y;
        // b matrix
        b.push_back(Mat1i({x_pr, y_pr}));
        // A matrix
        Mat1i m1(Mat1i({src_points[ii][0], src_points[ii][1], 1, 0, 0, 0,
                        -src_points[ii][0]*x_pr, -src_points[ii][1]*x_pr}).t());
                        
        Mat1i m2(Mat1i({0, 0, 0, src_points[ii][0], src_points[ii][1], 1,
                        -src_points[ii][0]*y_pr, -src_points[ii][1]*y_pr}).t());
        A.push_back(m1);
        A.push_back(m2);
    }
    
    // convert type
    A.convertTo(A, CV_32F);
    b.convertTo(b, CV_32F);
    
//    cout << A << endl;
//    cout << b.type() << endl;
    
    // Calculate H matrix
    Mat h = A.inv() * b;
//    cout << h << endl;
    
    float H_data[3][3] = {{h.at<float>(0), h.at<float>(1), h.at<float>(2)},
                          {h.at<float>(3), h.at<float>(4), h.at<float>(5)},
                          {h.at<float>(6), h.at<float>(7), 1}};
                           
    Mat H = Mat(3, 3, CV_32F, H_data);
//    cout << H << endl;

//    Pointi xxxxxx = calc_corresponding_point(-1871,
//                                             -1546,
//                                             1,
//                                             H.inv());
//    cout << xxxxxx.x << ' ' << xxxxxx.y << endl;                                         
    
    
    // Create blank image for source image
    float x_list[4];
    float y_list[4];
    for (int i=0; i<4; i++) {
        Pointi point_orig = calc_corresponding_point(corner_points[i].x,
                                                       corner_points[i].y,
                                                       1,
                                                       H);
//        cout << point_orig.x << ' ' << point_orig.y << endl;
        
        x_list[i] = point_orig.x;
        y_list[i] = point_orig.y;
        
    }
    
    float max_x = *max_element(x_list, x_list+4);
    float min_x = *min_element(x_list, x_list+4);
    float max_y = *max_element(y_list, y_list+4);
    float min_y = *min_element(y_list, y_list+4);
    
    float width  = max_x - min_x;
    float height = max_y - min_y;
//    cout << width << ' ' << height << endl;
    
    // For so large output images
    float aspect_ratio = width / height;
    int scaled_width = scaled_height * aspect_ratio;
    
    // The amount of size changing
    float w_change = width / scaled_width;
    float h_change = height / scaled_height;
    
    // Create a blank image for rectified image
    Mat rectified_img(scaled_height, scaled_width, CV_8UC3, Scalar(0, 0, 0));
    
    // Make rectified image: INVERSE method
    // Find color on source image by checking if the point is in ROI polygone    
    for (int i=0; i<rectified_img.cols; i++) {
        
        for (int j=0; j<rectified_img.rows; j++) {
            
            // Modify coordinates to find points smaller than ROI part
            float x_orig = (i * w_change) + min_x;
            float y_orig = (j * h_change) + min_y;
//            float x_orig = i + min_x;
//            float y_orig = j + min_y;
                
            // Find corressponding point on projected image
            Pointi rectified_point = calc_corresponding_point(x_orig,
                                                              y_orig,
                                                              1,
                                                              H.inv());
                                                                 
            if (point_in_polygon(rectified_point, corner_points)) {
                
                if (rectified_point.x > 0 & rectified_point.y > 0) {
                    
                    // Its color                                          
//                    Vec3b intensity = projected_img.at<Vec3b>((int)rectified_point.y,
//                                                              (int)rectified_point.x);
//                    float blue = intensity.val[0];
//                    float green = intensity.val[1];
//                    float red = intensity.val[2];
                    
                    // Use bilinear interpolation method to specify rectified color
                    colori est_color = color_estimator(rectified_point.x,
                                                       rectified_point.y,
                                                       projected_img);
                        
                    // Make rectified image
                    rectified_img.at<Vec3b>( j , i )[0] = est_color.blue;
                    rectified_img.at<Vec3b>( j , i )[1] = est_color.green;
                    rectified_img.at<Vec3b>( j , i )[2] = est_color.red;
                }
                
            } else {
                continue;
            }	        
            
        }
    }
    
    // Save rectified image
    imwrite(RESULT_PATH, rectified_img);
    cout << "Done!\n" << endl;
    return 0;
}






// Find corresponding point by using homography matrix
Pointi calc_corresponding_point(float x, float y, float z, Mat H) {
    
    Pointi result_point;
    float B_data[3][1] = {{x},
                          {y},
                          {z}};
    Mat B = Mat(3, 1, CV_32F, B_data);
    Mat xy_correspond = H * B;

    result_point.x = xy_correspond.at<float>(0,0)/xy_correspond.at<float>(0,2);
    result_point.y = xy_correspond.at<float>(0,1)/xy_correspond.at<float>(0,2);
    
    return result_point;
}



// Checking if a point is inside a polygon
bool point_in_polygon(Pointi point, vector<Pointi> polygon){

	int num_vertices = polygon.size();
	double x = point.x, y = point.y;
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
					double x_intersection
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


