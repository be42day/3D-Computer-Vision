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



tuple<float, float> calc_corresponding_point(float x, float y, Mat H);
bool point_in_polygon(Pointi point, vector<Pointi> polygon);
float distance(float point1[], float point2[]);
colori color_estimator(float x_coor, float y_coor, Mat image);



int main(int argc, char* argv[])
{
    
    string METHOD = argv[1];       // "direct" or "inverse"
    string TARGET_NAME = argv[2];  // "card1" or "card2" or "card3"
//    string HOMOGRAPHI = argv[3];
    string RESULT_PATH = "../images/results/result-on-" + TARGET_NAME + "-" + METHOD +".jpg";
    
    // Read target image (prime coordinates)
    string target_path = "../images/" + TARGET_NAME + ".jpeg";
    Mat target_img = imread(target_path, IMREAD_COLOR);
    
    // Read source image
    string source_path = "../images/al-najaf.jpg";
    Mat source_img = imread(source_path, IMREAD_COLOR);
//    cout << source_path.cols;
    
    // Show the image
//    imshow("Display window", source_img);
//    int k = waitKey(0);
    
    // Selected 4 points on the source image
    int points[4][2] = {
        {0, 0},
        {0, source_img.rows},
        {source_img.cols, source_img.rows},
        {source_img.cols, 0}
    };
    
    // Corresponding points on target image (PQRS)
    vector<Pointi> points_pr; 
    if (TARGET_NAME == "card1") {
    
        points_pr = {{ 528 , 285 }, { 632 , 1084 },
                     { 1210 , 790 }, { 1227 , 201 }
                    };
                    
    } else if (TARGET_NAME == "card2") {
    
        points_pr = {{ 324 , 250 }, { 224 , 842 },
                     { 852 , 1088 }, { 1008 , 260 }
                    };
                    
    } else {
    
        points_pr = {{ 585 , 80 }, { 94 , 588 },
                     { 702 , 1180 }, { 1194 , 676 }
                    };
    }
	
    // Homography matrix
    // To solve Ah = b equation
    
//    double H_data[3][3];
    int x_pr, y_pr;
    
//    if (HOMOGRAPHI == "projective") {
        
    Mat A;
    Mat b;
    for (int ii=0; ii<4; ii++){

        x_pr = points_pr[ii].x;
        y_pr = points_pr[ii].y;
        // b matrix
        b.push_back(Mat1i({x_pr, y_pr}));
        // A matrix
        Mat1i m1(Mat1i({points[ii][0], points[ii][1], 1, 0, 0, 0,
                        -points[ii][0]*x_pr, -points[ii][1]*x_pr}).t());
                        
        Mat1i m2(Mat1i({0, 0, 0, points[ii][0], points[ii][1], 1,
                        -points[ii][0]*y_pr, -points[ii][1]*y_pr}).t());
        A.push_back(m1);
        A.push_back(m2);
    }
    
    // convert type
    A.convertTo(A, CV_64F);
    b.convertTo(b, CV_64F);
    
//    cout << A << endl;
//    cout << b.type() << endl;
    
    // Calculate H matrix
    Mat h = A.inv() * b;
//    cout << h << endl;
    
    double H_data[3][3] = {{h.at<double>(0), h.at<double>(1), h.at<double>(2)},
                           {h.at<double>(3), h.at<double>(4), h.at<double>(5)},
                           {h.at<double>(6), h.at<double>(7), 1}};
                           
    Mat H = Mat(3, 3, CV_64F, H_data);   
    
//    cout << H << endl;

    // Transfer the image
    if (METHOD == "direct") {  
        
        // DIRECT method
        // For each point in source image, find its color and transfer the color
        // to corresponding point on target image  
        for (int i=0; i<source_img.cols; i++) {
            
            for (int j=0; j<source_img.rows; j++) {
                
                float x_orig = i;
                float y_orig = j;
                // Define a point to test
	            Pointi prime_point = { x_orig, y_orig };
	            
	            // Its color
	            Vec3b intensity = source_img.at<Vec3b>(y_orig, x_orig);
                float blue = intensity.val[0];
                float green = intensity.val[1];
                float red = intensity.val[2];
	            
	            // Find corresponding prime point
	            auto [x_prime, y_prime] = calc_corresponding_point(x_orig,
	                                                               y_orig, H);
		        
		        // Transfer the color
		        target_img.at<Vec3b>((int) y_prime, (int) x_prime)[0] = blue;
                target_img.at<Vec3b>((int) y_prime, (int) x_prime)[1] = green;
                target_img.at<Vec3b>((int) y_prime, (int) x_prime)[2] = red;
		        
            }
        }
    
    } else {
    
        // INVERSE method
        // Find color on source image by checking if the point is in ROI polygone    
        for (int i=0; i<target_img.cols; i++) {
            
            for (int j=0; j<target_img.rows; j++) {
                
                float x_prime = i;
                float y_prime = j;
                // Define a point to test
	            Pointi prime_point = { x_prime, y_prime };
	            
//	            cout << prime_point.x << ' ' << prime_point.y << endl;	        
		        
		        if (point_in_polygon(prime_point, points_pr)) {
		        
//		        cout << prime_point.x << ' ' << prime_point.y << endl;
        
//                    auto [x_orig, y_orig] = calc_invert_point(x_prime, y_prime, H.inv());
                    auto [x_orig, y_orig] = calc_corresponding_point(x_prime,
                                                                     y_prime,
                                                                     H.inv());
                    
                    if (x_orig > 0 && y_orig > 0) {                                                
//                    cout << x_orig << ' ' << y_orig << endl;
                    
                        // Bilinear Interpolation for color
                        colori est_color = color_estimator(x_orig, y_orig, source_img);
                        target_img.at<Vec3b>((int) y_prime, (int) x_prime)[0] = est_color.blue;
                        target_img.at<Vec3b>((int) y_prime, (int) x_prime)[1] = est_color.green;
                        target_img.at<Vec3b>((int) y_prime, (int) x_prime)[2] = est_color.red;
                    }
                    
                } else {
    //                cout << "incorrect point" << endl;
                    continue;
                }
		        
                
            }
        }
        
    }
    
    // Save result image
    imwrite(RESULT_PATH, target_img);
    cout << "Done!\n" << endl;
    return 0;
}







// Find corresponding point by using homography matrix
tuple<float, float> calc_corresponding_point(float x, float y, Mat H) {

    double B_data[3][1] = {{x},
                           {y},
                           {1}};
    Mat B = Mat(3, 1, CV_64F, B_data);
    Mat xy_correspond = H * B;
    xy_correspond.convertTo(xy_correspond, CV_32F);
    
    return {xy_correspond.at<float>(0,0)/xy_correspond.at<float>(0,2),
            xy_correspond.at<float>(0,1)/xy_correspond.at<float>(0,2)};
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





//    Mat ff(1, 8, CV_64F, 455,444,1);
//    Mat ff = Mat(3, 1, CV_64F, {455,444,1});
//    cout << h.reshape(3,3).type() << endl;
//    cout << h.type() << endl;
//    cout << ff.type() << endl;
    
//    Mat test = h * ff;
    
//    cout << test << endl;
//    XY_prime = XY_prime.t();
//    Mat XY = (Mat1i(3, 4) << 0, trg_img.cols, trg_img.cols, 0,
//                             0, 0, trg_img.rows, trg_img.rows,
//                             1 , 1, 1, 1);
//    
//    Mat H = XY.inv() * XY_prime;
//    cout << H << endl;
//    Mat XY = (Mat_(3, 4) << source_pts, 2, 3, 4);
    
    
//    Vec3b intensity = src_img.at<Vec3b>(y, x);
//    float blue = intensity.val[0];
//    float green = intensity.val[1];
//    float red = intensity.val[2];
//    cout << blue << endl << green << endl << red << endl;
    

