package com.example.scanner;

import static org.opencv.android.Utils.bitmapToMat;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.io.ByteArrayOutputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;

public class ScanActivity extends AppCompatActivity {

    //Scanned image display
    ImageView scannedImage;
    //Page number display
    TextView pageNum;

    //create mat object for image processing
    Mat matImg;
//    Mat theta;

    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_scan);
//get the image uri and pageNum passed from the previous activity
        scannedImage = findViewById(R.id.scanned);
        pageNum = findViewById(R.id.PageNum);

        Uri imageUri = getIntent().getExtras().getParcelable("URI");

//bytes array to store the image
        byte[] bytesImg = null;
        try {
            bytesImg = uriToBytes(imageUri);
//            Toast.makeText(this, "test msg", Toast.LENGTH_SHORT).show(); //test
        } catch (IOException e) {
//            throw new RuntimeException(e);
            Toast.makeText(this, "Fail to load image to bytes array.", Toast.LENGTH_SHORT).show(); //test

        }
        if (bytesImg == null) {
            Toast.makeText(this, "Error occurred when converting bytes array to mat object.", Toast.LENGTH_SHORT).show(); //test
        }

//bytes array converted to bitmap image
        Bitmap bitmap = BitmapFactory.decodeByteArray(bytesImg, 0, bytesImg.length); // test success
        Bitmap btmImg = bitmap.copy(Bitmap.Config.ARGB_8888, true);

//        Bitmap scaledBitmap = Bitmap.createScaledBitmap(btmImg, 800, 600, true);
        // Apply edge detection to the scaled bitmap
        Bitmap edgesBitmap = detectEdges(btmImg);

////bitmap to Mat object
        matImg = new Mat(btmImg.getHeight(), btmImg.getWidth(), CvType.CV_8UC3);
////        Toast.makeText(this, "new mat object obtained.", Toast.LENGTH_SHORT).show(); //test
        bitmapToMat(btmImg, matImg); // get mat object from the bitmap object
        if (matImg == null) {
            Toast.makeText(this, "No mat object obtained.", Toast.LENGTH_SHORT).show(); //test
        }

//////////////Scanner Algorithm applied here
        //canny edge detector
        double ratio = (double) matImg.rows() / 500.0;
        Mat copy = matImg.clone();
        Size newSize = new Size(matImg.cols() / ratio, 500);
        Imgproc.resize(matImg, matImg, newSize);
//        Toast.makeText(this, "test", Toast.LENGTH_SHORT).show(); //test


        //test // not used
        Mat test = new Mat();
//        Imgproc.GaussianBlur(gray, test, new Size(5, 5), 1.4);
//        Mat gray = new Mat();
//        Imgproc.cvtColor(matImg, gray, Imgproc.COLOR_BGR2GRAY);
//        Mat test = gaussian_blur(gray,5,1.4);
//        Mat test = sobel_filter(gray, "x");
//        Mat test = sobel_filter(gray, "y");
//        Mat test = edge_tracking(gray,gray);
//        Mat test = detectEdges(matImg, 75, 200);
//        Imgproc.Canny(matImg, test, 75, 200);
//        Toast.makeText(this, "test", Toast.LENGTH_SHORT).show(); //test
        //test done

//        Mat edges = new Mat();
        //edges obtained
//        Mat test = new Mat();
//        Mat test = canny_edge_detector(matImg,75,200);
///////////////Scanner Algorithm ended here

///////////////digit recognition Algorithm started here
//        int num_detected = digit_recognition(gray);
//        String test = Integer.toString(num_detected);

///////////////digit recognition Algorithm ended here

        //get the bitmap from the processed image
//        Bitmap scanned = Bitmap.createBitmap(test.cols(), test.rows(), Bitmap.Config.ARGB_8888);
        //convert the mat object back to bitmap
//        Utils.matToBitmap(test, scanned);

        //display the image from the gallery
//        scannedImage.setImageURI(imageUri);
//        scannedImage.setImageBitmap(scanned); // change to the scanned document later
        scannedImage.setImageBitmap(edgesBitmap);

        //Page Number Displayed
//        String text = null;
        String text = getIntent().getStringExtra("Page");
        if (text != null) {
            pageNum.setText(text);
        }
    }

    private Bitmap detectEdges(Bitmap inputBitmap) {
        int width = inputBitmap.getWidth();
        int height = inputBitmap.getHeight();
        Bitmap outputBitmap = Bitmap.createBitmap(width, height, inputBitmap.getConfig());

        int[][] Gx = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
        int[][] Gy = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

        int lowerThreshold = 75;
        int upperThreshold = 200;

        for (int y = 1; y < height - 1; y++) {
            for (int x = 1; x < width - 1; x++) {
                int sumX = 0;
                int sumY = 0;
                for (int k = -1; k <= 1; k++) {
                    for (int j = -1; j <= 1; j++) {
                        int color = inputBitmap.getPixel(x + j, y + k);
                        int grayscale = (Color.red(color) + Color.green(color) + Color.blue(color)) / 3;
                        sumX += Gx[k + 1][j + 1] * grayscale;
                        sumY += Gy[k + 1][j + 1] * grayscale;
                    }
                }

                int gradient = (int) Math.sqrt(sumX * sumX + sumY * sumY);

                if (gradient < lowerThreshold) {
                    gradient = 0;
                } else if (gradient > upperThreshold) {
                    gradient = 255;
                }

                int edgeColor = Color.rgb(gradient, gradient, gradient);
                outputBitmap.setPixel(x, y, edgeColor);
            }
        }
        return outputBitmap;
    }



    private byte[] uriToBytes(Uri uri) throws IOException {
        InputStream iStream = getContentResolver().openInputStream(uri);
        ByteArrayOutputStream byteArrayStream = new ByteArrayOutputStream();
        byte[] buffer = new byte[1024];

        int i;
        while ((i = iStream.read(buffer, 0, buffer.length)) > 0) {
            byteArrayStream.write(buffer, 0, i);
        }

        byte[] bytes = byteArrayStream.toByteArray();
        iStream.close();

        return bytes;
    }
//not used
//    private Mat detectEdges(Mat src, double lowThreshold, double highThreshold) {
//        Mat gray = new Mat();
//        Mat blur = new Mat();
//        Mat edges = new Mat();
//
//        // Convert the image to grayscale
//        Imgproc.cvtColor(src, gray, Imgproc.COLOR_BGR2GRAY);
//        // Apply Gaussian blur to reduce noise and smooth the image
//        Imgproc.GaussianBlur(gray, blur, new Size(5, 5), 0); // can also use
//        // Apply the custom Canny edge detection
//        applyCanny(blur, edges, lowThreshold, highThreshold);
//
//        return edges;
//    }
//    private void applyCanny(Mat src, Mat dst, double lowThreshold, double highThreshold) {
//        final int CANNY_SHIFT = 16;
//        final double CANNY_SCALE = 1 << CANNY_SHIFT;
//
//        Mat mag = new Mat();
//        Mat gradient = new Mat();
//
//        // Calculate gradient and magnitude
//        Imgproc.Sobel(src, gradient, CvType.CV_16S, 1, 0);
//        Core.convertScaleAbs(gradient, mag);
//
//        // Create an edge map
//        Mat edgeMap = Mat.zeros(src.size(), CvType.CV_8UC1);
//
//        for (int y = 1; y < src.rows() - 1; y++) {
//            for (int x = 1; x < src.cols() - 1; x++) {
//                int m = (int) (mag.get(y, x)[0] * CANNY_SCALE);
//
//                if (m > lowThreshold * CANNY_SCALE) {
//                    boolean isEdge = m > highThreshold * CANNY_SCALE;
//
//                    for (int dy = -1; dy <= 1; dy++) {
//                        for (int dx = -1; dx <= 1; dx++) {
//                            if (isEdge || (int) (mag.get(y + dy, x + dx)[0] * CANNY_SCALE) > m) {
//                                edgeMap.put(y, x, 255);
//                                isEdge = true;
//                                break;
//                            }
//                        }
//                        if (isEdge) {
//                            break;
//                        }
//                    }
//                }
//            }
//        }
//        dst.release();
//        dst = edgeMap.clone();
//    }

    //This function ended up not being used cuz it kept crashing the app.
    private Mat canny_edge_detector(Mat image, double lowThresholdRatio, double highThresholdRatio) {
        //grayscale the image
        Mat gray = new Mat();
        Imgproc.cvtColor(image, gray, Imgproc.COLOR_BGR2GRAY);
        //Gaussian blur the image
        Mat img = gaussian_blur(gray, 5,1.4);
        //Compute the gradient of the image using Sobel operator
        Mat Gx = sobel_filter(img, "x");
        Mat Gy = sobel_filter(img, "y");
        //Compute the magnitude and direction of gradient
        Mat magnitude = new Mat();
        Mat angle = new Mat();
        Core.magnitude(Gx, Gy, magnitude);
        Core.phase(Gx, Gy, angle,true);

        // Perform non-maximum suppression
        Mat nms = non_maximum_suppression(magnitude, angle);

        //setting the minimum and maximum thresholds for double thresholding
        double mag_max = Core.minMaxLoc(magnitude).maxVal;
        double lowThreshold = lowThresholdRatio == 0 ? mag_max * 0.1 : mag_max * lowThresholdRatio;
        double highThreshold = highThresholdRatio == 0 ? mag_max * 0.5 : mag_max * highThresholdRatio;

        Mat weak_edges = Mat.zeros(image.size(), image.type());
        Mat strong_edges = Mat.zeros(image.size(), image.type());
        Mat edges = Mat.zeros(image.size(), image.type());

        for (int i = 0; i < nms.rows(); i++) {
            for (int j = 0; j < nms.cols(); j++) {
                double[] pixel = nms.get(i, j);
                if (pixel[0] >= highThreshold) {
                    double[] newVal = {255};
                    strong_edges.put(i, j, newVal);
                } else if (pixel[0] < lowThreshold) {
                    double[] newVal = {0};
                    edges.put(i, j, newVal);
                } else {
                    double[] newVal = {1};
                    weak_edges.put(i, j, newVal);
                }
            }
        }

        // Call edge_tracking function
        Mat tracked_edges = edge_tracking(weak_edges, strong_edges);
        Core.add(edges, tracked_edges, edges);

        return edges;
    }

    //These function works perfectly but not used
    private Mat gaussian_kernel(int size, double sigma) {
        Mat kernel = new Mat(size, size, CvType.CV_64F);
        int center = size / 2;
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                double x = i - center;
                double y = j - center;
                double value = Math.exp(-(x * x + y * y) / (2 * sigma * sigma));
                kernel.put(i, j, value);
            }
        }
        // Normalize the kernel
        Core.divide(kernel, new Scalar(2 * Math.PI * sigma * sigma), kernel);
        return kernel;
    }
    private Mat gaussian_blur(Mat image, int size, double sigma) {
        Mat kernel = gaussian_kernel(size, sigma);
        Mat filteredImage = Mat.zeros(image.size(),image.type());
        Imgproc.filter2D(image, filteredImage, -1, kernel);
        return filteredImage;
    }
    private Mat sobel_filter(Mat img, String axis) {
        int dx;
        int dy;
        if (axis.equals("x")) {
            dx = 1;
            dy = 0;
        } else if (axis.equals("y")) {
            dx = 0;
            dy = 1;
        } else {
            throw new IllegalArgumentException("Invalid axis parameter. Must be 'x' or 'y'.");
        }
        Mat output = new Mat(img.rows(),img.cols(),img.type());
        Imgproc.Sobel(img, output, img.type(), dx, dy);
        return output;
    }

    //Not sure if this function is functional, hard to test it, but 90% sure its working
    private Mat edge_tracking(Mat img_weak, Mat img_strong) {
        int height = img_weak.rows();
        int width = img_weak.cols();

        for (int i = 1; i < width - 1; i++) {
            for (int j = 1; j < height - 1; j++) {
                double weakPixel = img_weak.get(j, i)[0];
                if (weakPixel != 0) {
                    Mat submat = img_strong.submat(j - 1, j + 2, i - 1, i + 2);
                    double maxVal = Core.minMaxLoc(submat).maxVal;
                    if (maxVal != 0) {
                        img_strong.put(j, i, 1);
                    }
                }
            }
        }
        return img_strong;
    }

    //Not used due to the change of method
    private Mat non_maximum_suppression(Mat magnitude, Mat angle) {
        Mat output = Mat.zeros(magnitude.size(), magnitude.type());
        int height = magnitude.rows();
        int width = magnitude.cols();

        for (int i = 1; i < height - 1; i++) {
            for (int j = 1; j < width - 1; j++) {
                double value = magnitude.get(i, j)[0];
                double angle_val = angle.get(i, j)[0];
                angle_val = angle_val < 0 ? angle_val + 180 : angle_val;
                int q = 255;
                int r = 255;
                // Determine the appropriate neighbors to compare against based on the gradient angle
                if ((0 <= angle_val && angle_val < 22.5) || (157.5 <= angle_val && angle_val <= 180)) {
                    q = (int) magnitude.get(i, j + 1)[0];
                    r = (int) magnitude.get(i, j - 1)[0];
                } else if (22.5 <= angle_val && angle_val < 67.5) {
                    q = (int) magnitude.get(i + 1, j + 1)[0];
                    r = (int) magnitude.get(i - 1, j - 1)[0];
                } else if (67.5 <= angle_val && angle_val < 112.5) {
                    q = (int) magnitude.get(i + 1, j)[0];
                    r = (int) magnitude.get(i - 1, j)[0];
                } else if (112.5 <= angle_val && angle_val < 157.5) {
                    q = (int) magnitude.get(i - 1, j + 1)[0];
                    r = (int) magnitude.get(i + 1, j - 1)[0];
                }
                // Perform non-maximum suppression
                if (value >= q && value >= r) {
                    double[] newVal = {value};
                    output.put(i, j, newVal);
                } else {
                    double[] newVal = {0};
                    output.put(i, j, newVal);
                }
            }
        }
        return output;
    }

}
