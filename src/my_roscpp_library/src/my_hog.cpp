#ifndef MY_HOG_CPP
#define MY_HOG_CPP

#include <my_roscpp_library/my_hog.h>
#include <ros/ros.h>


int  vCount = 0, vCounter=0;
int vFile = 50;

using namespace std;
using namespace cv;

//Mat featureVec(vFile, 9, CV_32F);
Mat featureVecOne(1, 9, CV_32F);
Mat featureVecFull(vFile, 9 * 16, CV_32F);
Mat featureVecHOG(1, 9 * 16, CV_32F);
Mat vecHOGGLCM(2, vFile * 6, CV_32F);
vector< vector <float> > v_descriptorsValues;
vector< vector <float> > v_desHOGglcmValues;
vector< vector <float> > v_desEigenValues;
vector<float> vecGlcm;
vector<float> vecHOG_GLCM;
vector<float> newHOGVec;
vector<float> VecEigenValues;
Mat newReduced;
PCA pca1;

void initVFile(int n) {
  vFile = n;
  //cout << "n: " << n << " || vFile: " << vFile << endl;
}

/// function to compute angle and magnitude image
void computeMagAngle(InputArray src, OutputArray mag, OutputArray ang)
{
  //cout << ">" << "computeMagAngle" << endl;
  Mat img = src.getMat();
  img.convertTo(img, CV_32F, 1 / 255.0);

  /// calculate gradients using sobel
  Mat gx, gy;
  Sobel(img, gx, CV_32F, 1, 0, 1);
  Sobel(img, gy, CV_32F, 0, 1, 1);

  /// Calculate gradient magnitude and direction
  Mat magnitude, angle;
  cartToPolar(gx, gy, mag, ang, 1);
}


/// function to compute histogram of oriented gradients feature
void computeHOG(InputArray mag, InputArray ang, OutputArray dst, int dims, bool isWeighted = true)
{
  //cout << ">" << "computeHOG" << endl;
  /// init input values
  Mat magMat = mag.getMat();
  Mat angMat = ang.getMat();

  /// validate magnitude and angle dimensions
  if (magMat.rows != angMat.rows || magMat.cols != angMat.cols) {
    return;
  }

  /// get row and col dimensions
  int rows = magMat.rows;
  int cols = magMat.cols;

  /// set up the expected feature dimension, and
  /// compute the histogram bin length (arc degree)
  int featureDim = dims;
  float circleDegree = 360.0;
  float binLength = circleDegree / (float)featureDim;
  float halfBin = binLength / 2;

  /// set up the output feature vector
  /// upper limit and median for each bin
  //featureVec = 0.0;
  featureVecOne = 0.0;
  vector<float> uplimits(featureDim);
  vector<float> medbins(featureDim);

  for (int i = 0; i < featureDim; i++) {
    uplimits[i] = (2 * i + 1) * halfBin;
    medbins[i] = i * binLength;

    //cout << "[" << medbins[i] << "] ";
    //cout << uplimits[i] << " ";
  }
  /*cout << endl;
  cout << "[x] x" << " -> [Medbins[i]] Uplimit[i]";
  cout << endl;*/

  /// begin calculate the feature vector
  for (int i = 0; i < rows; i++)
  {
    for (int j = 0; j < cols; j++)
    {
      /// get the value of angle and magnitude for
      /// the current index (i,j)
      float angleVal = angMat.at<float>(i, j);
      float magnitudeVal = magMat.at<float>(i, j);

      /// (this is used to calculate weights)
      float dif = 0.0; /// dfference between the angle and the bin value
      float prop = 0.0; /// proportion for the value of the current bin

                /// value to add for the histogram bin of interest
      float valueToAdd = 0.0;
      /// value to add for the neighbour of histogram bin of interest
      float sideValueToAdd = 0.0;
      /// index for the bin of interest and the neighbour
      int valueIdx = 0;
      int sideIdx = 0;

      /// the first bin (zeroth index) is a little bit tricky
      /// because its value ranges between below 360 degree and higher 0 degree
      /// we need something more intelligent approach than this
      if (angleVal <= uplimits[0] || angleVal >= uplimits[featureDim - 1]) {
        if (!isWeighted) {
          //featureVec.at<float>(vCounter, 0) += magnitudeVal;
          featureVecOne.at<float>(0, 0) += magnitudeVal;
        }
        else {
          if (angleVal >= medbins[0] && angleVal <= uplimits[0]) {
            dif = abs(angleVal - medbins[0]);

            valueIdx = 0;
            sideIdx = 1;
          }
          else {
            dif = abs(angleVal - circleDegree);

            valueIdx = 0;
            sideIdx = featureDim - 1;
          }
        }

      }
      /// this is for the second until the last bin
      else {
        for (int k = 0; k < featureDim - 1; k++) {
          if (angleVal >= uplimits[k] && angleVal < uplimits[k + 1]) {
            if (!isWeighted) {
              //featureVec.at<float>(vCounter, k + 1) += magnitudeVal;
              featureVecOne.at<float>(0, k + 1) += magnitudeVal;
            }
            else {
              dif = abs(angleVal - medbins[k + 1]);
              valueIdx = k + 1;

              if (angleVal >= medbins[k + 1]) {
                sideIdx = (k + 1 == featureDim - 1) ? 0 : k + 2;
              }
              else {
                sideIdx = k;
              }
            }
            break;
          }
        }
      }

      /// add the value proportionally depends of
      /// how close the angle to the median limits
      if (isWeighted) {
        prop = (binLength - dif) / binLength;
        valueToAdd = prop * magnitudeVal;
        sideValueToAdd = (1.00 - prop) * magnitudeVal;
        //featureVec.at<float>(vCounter, valueIdx) += valueToAdd;
        //featureVec.at<float>(vCounter, sideIdx) += sideValueToAdd;
        /// add one vector
        featureVecOne.at<float>(0, valueIdx) += valueToAdd;
        featureVecOne.at<float>(0, sideIdx) += sideValueToAdd;
      }
      /*cout << endl;
      cout << "angleVal: " << angleVal << "\t| valueIdx: " << valueIdx << "\t| sideIdx: " << sideIdx << endl;
      cout << "binLength: " << binLength << "\t| dif: " << dif << "\t| prop: " << prop << endl;
      cout << "binLength-dif: " << binLength - dif << "\t\t| (binLength-dif)/binLength: " << (binLength - dif) / binLength << endl;
      cout << "ValueToAdd: " << valueToAdd << "\t\t| SideValueToAdd: " << sideValueToAdd << endl;
      cout << "featureVec -> ";
      for (int i = 0; i < featureDim; i++) cout << featureVec.at<float>(0, i) << " " ;
      cout << endl;
      count++;*/
    }

  }
  //for (int y = 0; y < 9; y++) {
  //	featureVecFull.at<float>(vCounter, y + vCount) = featureVec.at<float>(vCounter, y);
  //}
  for (int y = 0; y < 9; y++) {
    featureVecHOG.at<float>(0, y + vCount) = featureVecOne.at<float>(0, y);
  }
  vCount += 9;
}

void featureVecFullPrint(vector<float>& vHOG, int loop, bool print) {
  if(print == true) cout << "> " << "Begin feature vec full..." << endl;
  /// Print FeatureVecFull
  /*for (int i = 0; i < vFile; i++) {
    for (int j = 0; j < 9 * 16; j++) {
      cout << featureVecFull.at<float>(i, j) << " ";
    }
  }*/
  //for (int y = 0; y < 9 * 16; y++) {
  //	vHOG.push_back(featureVecFull.at<float>(loop, y));
  //}
  for (int y = 0; y < 9 * 16; y++) {
    newHOGVec.push_back(featureVecHOG.at<float>(0, y));
  }
  Mat featureVecOne(1, 9, CV_32F);
  Mat featureVecHOG(1, 9 * 16, CV_32F);
  v_descriptorsValues.push_back(vHOG);

  //for (auto i = vHOG.begin(); i != vHOG.end(); ++i) cout << *i << " ";
  //for (auto i = newHOGVec.begin(); i != newHOGVec.end(); ++i) cout << *i << " ";

  vCount = 0;
  vCounter++;
  if (print == true) {
    cout << ">" << "Print featureVecFull...";
    cout << endl << "Print vHOG: " << loop << endl;
    for (auto i = vHOG.begin(); i != vHOG.end(); ++i) cout << *i << " ";
    cout << endl << "--------------------------------------------------------<" << loop << ">----------------------------------------------------------" << endl << endl;
  }
}

void reduceFeatureUsingPCA(Mat reduced, int maxComp, bool isPrint) {
  cout << "\nBegin reducing feature HOG using PCA..." << endl;

  PCA pca(featureVecFull, Mat(), CV_PCA_DATA_AS_ROW, maxComp);
  reduced = pca.project(featureVecFull);
  cout << "from: " << featureVecFull.size() << " to ";
  cout << reduced.size() << endl;
  cout << "[col x row] " << endl;
  int id = 0;
  float intensity = 0;
  for (int i = 0; i < reduced.rows; i++) {
    if(isPrint == true) cout << "image " << i << ":\t";
    for (int j = 0; j < reduced.cols; j++) {
      intensity = reduced.at<float>(i, j);
      if (isPrint == true) cout << "[" << intensity << "]" << "\t";
      vecHOGGLCM.at<float>(0, id++) = intensity;
    }
    if (isPrint == true) cout << endl;
  }
  Mat eigenValue = pca.eigenvalues;
  Mat eigenVector = pca.eigenvectors;
  Mat mean = pca.mean;

  //cout << endl << "Eigen Value" << endl;
  for (int i = 0; i < eigenValue.rows; i++) {
    for (int j = 0; j < eigenValue.cols; j++) {
      VecEigenValues.push_back(eigenValue.at<float>(i, j));
      float inten = eigenValue.at<float>(i, j);
      //cout << inten << " ";
    }
    //cout << endl << endl;
  }
  //cout << endl << "Eigen Vector" << endl;
  for (int i = 0; i < eigenVector.rows; i++) {
    for (int j = 0; j < eigenVector.cols; j++) {
      VecEigenValues.push_back(eigenVector.at<float>(i, j));
      float inten = eigenVector.at<float>(i, j);
      //cout << inten << " ";
    }
    //cout << endl << endl;
  }
  //cout << endl << "Mean" << endl;
  for (int i = 0; i < mean.rows; i++) {
    for (int j = 0; j < mean.cols; j++) {
      VecEigenValues.push_back(mean.at<float>(i, j));
      float inten = mean.at<float>(i, j);
      //cout << inten << " ";
    }
    //cout << endl << endl;
  }

  //for (auto i = VecEigenValues.begin(); i != VecEigenValues.end(); ++i) cout << *i << " ";

  v_desEigenValues.push_back(VecEigenValues);

  cout << "eigenValue: " << eigenValue.size() << endl;
  for (int i = 0; i < eigenValue.rows; i++) {
    for (int j = 0; j < eigenValue.cols; j++) {
      float in = eigenValue.at<float>(i, j);
      cout << in << " ";
    }
    cout << endl;
  }
  cout << "eigenVector: " << eigenVector.size() << endl;
  /*for (int i = 0; i < eigenVector.rows; i++) {
    for (int j = 0; j < eigenVector.cols; j++) {
      float in = eigenVector.at<float>(i, j);
      cout << in << " ";
    }
    cout << endl << endl << endl;
  }*/

}

void reduceFeatureUsingPCAinSVM(Mat reduced, Mat Output, vector<float>& vectorHogGlcm, bool isPrint) {
  if (isPrint == true) cout << "\nBegin reducing feature HOG using PCA..." << endl;
  Mat eigenValue(1,6,CV_32F); Mat eigenVector(6,144,CV_32F) ; Mat mean(1, 144, CV_32F);
  PCA pca;
  int x=0, y=0;

  /// EigenValuesData
  for (int i = 0; i < 6; i++) {
    eigenValue.at<float>(0,i) = reduced.at<float>(0, i);
  }
  /// EigenVectorData
  for (int i = 6; i < 870; i++) {
    eigenVector.at<float>(x, y) = reduced.at<float>(0,i);
    y++;
    if (y == 144) {
      //cout << "x: " << x << " y: " << y << endl;
      x++; y = 0;
    }
  }
  /// Mean Data
  for (int i = 870; i < 1014; i++) {
    mean.at<float>(0,i-870) = reduced.at<float>(0, i);
  }

  if (isPrint == true) {
    cout << "eigenValue: " << eigenValue.size() << endl;
    cout << "eigenVector: " << eigenVector.size() << endl;
    cout << "mean: " << mean.size() << endl;
  }

  pca1.eigenvalues = eigenValue;
  pca1.eigenvectors = eigenVector;
  pca1.mean = mean;
  Output = pca1.project(newHOGVec);
  int id = 0;
  float intensity = 0;
  for (int i = 0; i < Output.rows; i++) {
    if (isPrint == true) cout << "image " << i << ":\t";
    for (int j = 0; j < Output.cols; j++) {
      intensity = Output.at<float>(i, j);
      if (isPrint == true) cout << "[" << intensity << "]" << "\t";
      vectorHogGlcm.push_back(intensity);
    }
    if (isPrint == true) cout << endl;
  }

  newHOGVec.clear();
}

void copyHOG_GLCMtoVec(vector<float>& vecGlcm, vector< vector <float>>& vecForSvm) {
  int id = 0;
  for (auto i = vecGlcm.begin(); i != vecGlcm.end(); ++i) vecHOGGLCM.at<float>(1, id++) = *i ;
  for (int i = 0; i < vFile; i++) {
    vecHOG_GLCM.clear();
    for (int j = 0; j < 6; j++) {
      vecHOG_GLCM.push_back(vecHOGGLCM.at<float>(0, (6*i)+j));
    }
    for (int k = 0; k < 6; k++) {
      vecHOG_GLCM.push_back(vecHOGGLCM.at<float>(1, (i*6)+k));
    }
    v_desHOGglcmValues.push_back(vecHOG_GLCM);
    vecForSvm.push_back(vecHOG_GLCM);
  }
  /// Print check data
  cout << "Print vecHOG_GLCM" << endl;
  for (int i = 0; i < vecHOGGLCM.rows; i++) {
    for (int j = 0; j < vecHOGGLCM.cols; j++) {
      float in = vecHOGGLCM.at<float>(i, j);
      cout << in << " ";
    }
    cout << endl << endl;
  }

  /*cout << "Print vecHOG_GLCM" << endl;
  for (auto i = vecHOG_GLCM.begin(); i != vecHOG_GLCM.end(); ++i) cout << *i << " ";

  int a = 0;
  for (const auto &row : v_desHOGglcmValues)
  {
    cout << endl << endl;
    cout << "row: " << ++a << endl << "---------------------------------------------------" << endl;
    int i = 0;
    for (const auto &s : row) {
      cout << ++i << "-> " << s << " ";
    }
    cout << endl << endl << endl;
  }*/
}

void saveEigenValues(char SaveHogDesFileName[100]) {
  //save to xml
  cout << "Save to: " << SaveHogDesFileName << endl;
  FileStorage hogXml(SaveHogDesFileName, FileStorage::WRITE); //FileStorage::READ
                                //2d vector to Mat
  int row = v_desEigenValues.size(), col = v_desEigenValues[0].size();
  printf("col=%d, row=%d\n", col, row);
  Mat M(row, col, CV_32F);
  //save Mat to XML
  for (int i = 0; i< row; ++i)
    memcpy(&(M.data[col * i * sizeof(float)]), v_desEigenValues[i].data(), col * sizeof(float));
  //write xml
  write(hogXml, "Descriptor_of_images", M);
  hogXml.release();
}

void saveFeatureVecFull(char SaveHogDesFileName[100]) {
  //save to xml
  cout << "Save to: " << SaveHogDesFileName << endl;
  FileStorage hogXml(SaveHogDesFileName, FileStorage::WRITE); //FileStorage::READ
                                //2d vector to Mat
  int row = v_descriptorsValues.size(), col = v_descriptorsValues[0].size();
  printf("col=%d, row=%d\n", col, row);
  Mat M(row, col, CV_32F);
  //save Mat to XML
  for (int i = 0; i< row; ++i)
    memcpy(&(M.data[col * i * sizeof(float)]), v_descriptorsValues[i].data(), col * sizeof(float));
  //write xml
  write(hogXml, "Descriptor_of_images", M);
  hogXml.release();
}

void saveHOGglcmVec(char SaveHogglcmDesFileName[100]) {
  //save to xml
  cout << "Save to: " << SaveHogglcmDesFileName << endl;
  FileStorage hogXml(SaveHogglcmDesFileName, FileStorage::WRITE); //FileStorage::READ
                                //2d vector to Mat
  int row = v_desHOGglcmValues.size(), col = v_desHOGglcmValues[0].size();
  printf("col=%d, row=%d\n", col, row);
  Mat M(row, col, CV_32F);
  //save Mat to XML
  for (int i = 0; i< row; ++i)
    memcpy(&(M.data[col * i * sizeof(float)]), v_desHOGglcmValues[i].data(), col * sizeof(float));
  //write xml
  write(hogXml, "Descriptor_of_images", M);
  hogXml.release();
}

/// HOG Visualisation
Mat get_hogdescriptor_visual_image(Mat& origImg,
  vector< float>& descriptorValues,
  Size winSize,
  Size cellSize,
  int scaleFactor,
  double viz_factor)
{
  Mat visual_image;
  resize(origImg, visual_image, Size(origImg.cols*scaleFactor, origImg.rows*scaleFactor));
  cvtColor(visual_image, visual_image, CV_GRAY2BGR);


  int gradientBinSize = 9;
  // dividing 180� into 9 bins, how large (in rad) is one bin?
  float radRangeForOneBin = 3.14 / (float)gradientBinSize;

  // prepare data structure: 9 orientation / gradient strenghts for each cell
  int cells_in_x_dir = winSize.width / cellSize.width;
  int cells_in_y_dir = winSize.height / cellSize.height;
  int totalnrofcells = cells_in_x_dir * cells_in_y_dir;
  float*** gradientStrengths = new float**[cells_in_y_dir];
  int** cellUpdateCounter = new int*[cells_in_y_dir];
  for (int y = 0; y< cells_in_y_dir; y++)
  {
    gradientStrengths[y] = new float*[cells_in_x_dir];
    cellUpdateCounter[y] = new int[cells_in_x_dir];
    for (int x = 0; x< cells_in_x_dir; x++)
    {
      gradientStrengths[y][x] = new float[gradientBinSize];
      cellUpdateCounter[y][x] = 0;

      for (int bin = 0; bin< gradientBinSize; bin++)
        gradientStrengths[y][x][bin] = 0.0;
    }
  }
  /*
  // nr of blocks = nr of cells - 1
  // since there is a new block on each cell (overlapping blocks!) but the last one
  int blocks_in_x_dir = cells_in_x_dir - 1;
  int blocks_in_y_dir = cells_in_y_dir - 1;

  // compute gradient strengths per cell
  int descriptorDataIdx = 0;
  int cellx = 0;
  int celly = 0;

  for (int blockx = 0; blockx< blocks_in_x_dir; blockx++)
  {
    for (int blocky = 0; blocky< blocks_in_y_dir; blocky++)
    {
      // 4 cells per block ...
      for (int cellNr = 0; cellNr< 4; cellNr++)
      {
        // compute corresponding cell nr
        int cellx = blockx;
        int celly = blocky;
        if (cellNr == 1) celly++;
        if (cellNr == 2) cellx++;
        if (cellNr == 3)
        {
          cellx++;
          celly++;
        }

        for (int bin = 0; bin< gradientBinSize; bin++)
        {
          float gradientStrength = descriptorValues[descriptorDataIdx];
          descriptorDataIdx++;

          gradientStrengths[celly][cellx][bin] += gradientStrength;

        } // for (all bins)


          // note: overlapping blocks lead to multiple updates of this sum!
          // we therefore keep track how often a cell was updated,
          // to compute average gradient strengths
        cellUpdateCounter[celly][cellx]++;

      } // for (all cells)


    } // for (all block x pos)
  } // for (all block y pos)


    // compute average gradient strengths
  for (int celly = 0; celly< cells_in_y_dir; celly++)
  {
    for (int cellx = 0; cellx< cells_in_x_dir; cellx++)
    {

      float NrUpdatesForThisCell = (float)cellUpdateCounter[celly][cellx];

      // compute average gradient strenghts for each gradient bin direction
      for (int bin = 0; bin< gradientBinSize; bin++)
      {
        gradientStrengths[celly][cellx][bin] /= NrUpdatesForThisCell;
      }
    }
  }


  cout << "descriptorDataIdx = " << descriptorDataIdx << endl;*/

  // draw cells
  for (int celly = 0; celly< cells_in_y_dir; celly++)
  {
    for (int cellx = 0; cellx< cells_in_x_dir; cellx++)
    {
      int drawX = cellx * cellSize.width;
      int drawY = celly * cellSize.height;

      int mx = drawX + cellSize.width / 2;
      int my = drawY + cellSize.height / 2;

      rectangle(visual_image,
        Point(drawX*scaleFactor, drawY*scaleFactor),
        Point((drawX + cellSize.width)*scaleFactor,
        (drawY + cellSize.height)*scaleFactor),
        CV_RGB(100, 100, 100),
        1);

      // draw in each cell all 9 gradient strengths
      for (int bin = 0; bin< gradientBinSize; bin++)
      {
        float currentGradStrength = gradientStrengths[celly][cellx][bin];

        // no line to draw?
        if (currentGradStrength == 0)
          continue;

        float currRad = bin * radRangeForOneBin + radRangeForOneBin / 2;

        float dirVecX = sin(currRad);
        float dirVecY = cos(currRad);
        float maxVecLen = cellSize.width / 2;
        float scale = viz_factor; // just a visual_imagealization scale,
                      // to see the lines better

                      // compute line coordinates
        float x1 = mx - dirVecX * currentGradStrength * maxVecLen * scale;
        float y1 = my - dirVecY * currentGradStrength * maxVecLen * scale;
        float x2 = mx + dirVecX * currentGradStrength * maxVecLen * scale;
        float y2 = my + dirVecY * currentGradStrength * maxVecLen * scale;

        // draw gradient visual_imagealization
        line(visual_image,
          Point(x1*scaleFactor, y2*scaleFactor),
          Point(x2*scaleFactor, y1*scaleFactor),
          CV_RGB(0, 0, 255),
          1);

      } // for (all bins)

    } // for (cellx)
  } // for (celly)


    // don't forget to free memory allocated by helper data structures!
  for (int y = 0; y< cells_in_y_dir; y++)
  {
    for (int x = 0; x< cells_in_x_dir; x++)
    {
      delete[] gradientStrengths[y][x];
    }
    delete[] gradientStrengths[y];
    delete[] cellUpdateCounter[y];
  }
  delete[] gradientStrengths;
  delete[] cellUpdateCounter;

  return visual_image;

}


void ini_coba()
{
    ROS_INFO("Coba berhasil!");
}

#endif // MY_HOG_CPP
