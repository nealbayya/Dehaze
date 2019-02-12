#include <bits/stdc++.h>
#include <opencv2/opencv.hpp>
using namespace std;
#define ROWS 370
#define COLS 425
#define FMAX 64 // COLS/4
#define SZ 2
#define WS 5 // 2*SZ+1 = conv size
#define WDTH 421 // COLS-2*SZ
#define INF 100000.0

void toArray(float (&dst)[ROWS][COLS][3], cv::Mat &src) {
   for (int i=0; i<ROWS; i++)
      for (int j=0; j<COLS; j++) {
         cv::Vec3b& color = src.at<cv::Vec3b>(i, j);
         for (int k=0; k<3; k++)
            dst[i][j][k] = color[k];
      }
}

void toImage(float (&src)[ROWS][COLS][3], cv::Mat dst) {
   for (int i=0; i<ROWS; i++)
      for (int j=0; j<COLS; j++) {
         cv::Vec3b& color = dst.at<cv::Vec3b>(i, j);
         for (int k=0; k<3; k++)
            color[k] = src[i][j][k];
      }
}

void norm(float (&img)[ROWS][COLS]) {
   float mx = 0;
   for (int i=0; i<ROWS; i++)
      for (int j=0; j<COLS; j++)
         mx = max(mx, img[i][j]);
   for (int i=0; i<ROWS; i++)
      for (int j=0; j<COLS; j++)
         img[i][j]=max(0.0, min(255.0, img[i][j]*255.0));
}

// Statistics Functions
float mean(float (&img)[ROWS][COLS][3], int r, int c, int ch) {
   float sum = 0;
   for (int i=r; i<r+WS; i++)
      for (int j=c; j<c+WS; j++)
         sum+=img[i][j][ch];
   return sum/(WS*WS);
}

float stdv(float (&img)[ROWS][COLS][3], float m, int r, int c, int ch) {
   float sum = 0;
   for (int i=r; i<r+WS; i++)
      for (int j=c; j<c+WS; j++)
         sum+=pow(img[i][j][ch]-m, 2);
   return sqrt(sum/(WS*WS));
}

float dncc(float (&img_a)[ROWS][COLS][3], float (&img_b)[ROWS][COLS][3], int ra, int ca, int rb, int cb, int ch) {
   float p1m, p2m, product;
   p1m = mean(img_a, ra, ca, ch);
   p2m = mean(img_b, rb, cb, ch);
   for (int i=0; i<WS; i++)
      for (int j=0; j<WS; j++)
         product+=(img_a[ra+i][ca+j][ch]-p1m)*(img_b[rb+i][cb+j][ch]-p2m);
   product/=(WS*WS);
   float stds = stdv(img_a, p1m, ra, ca, ch)*stdv(img_b, p2m, rb, cb, ch);
   if (stds == 0)
      return 0;
   product/=stds;
   return 0.5*(1.0-product);
}

// Stereo Algorithm
float cost[ROWS][WDTH+1][FMAX+1];
float costblur[ROWS][WDTH+1][FMAX+1];
float T = 240;
void computeCost(float (&img_a)[ROWS][COLS][3], float (&img_b)[ROWS][COLS][3], int r) {
   for (int i=0; i<=WDTH; i++)
      for (int j=0; j<=FMAX; j++) { // col1 = s+i, col2 = min(s+i+j, COLS-s)
         if (i==0 || j==0)
            continue;
         if (2*SZ+i+j>COLS)
            continue;
         cost[r][i][j] = dncc(img_a, img_b, r-SZ, i, r-SZ, i+j, 0)+
                         dncc(img_a, img_b, r-SZ, i, r-SZ, i+j, 1)+
                         dncc(img_a, img_b, r-SZ, i, r-SZ, i+j, 2);
         cost[r][i][j]/=3.0;
      }
}

#define RW 3
#define CW 1
float gr[2*RW+1] = {0.106595, 0.140367, 0.165569, 0.174938, 0.165569, 0.140367, 0.106595};
float gc[2*CW+1] = {0.319466, 0.361069, 0.319466};

void blur(int r) { // seperable 2D convolution
   float temp[WDTH+1][FMAX+1];
   for (int k=0; k<=FMAX; k++) {
      for (int j=0; j<=WDTH; j++) {
         float sum = 0;
         for (int i=-RW; i<=RW; i++) {
            int cr = r+i;
            if (cr<0 || cr>=ROWS)
               continue;
            sum+=gr[i+RW]*cost[cr][j][k];
         }
         temp[j][k] = sum;
      }
      for (int j=0; j<=WDTH; j++) {
         float sum = 0;
         for (int i=-CW; i<=CW; i++) {
            int cc = j+i;
            if (cc<0 || cc>WDTH)
               continue;
            sum+=gc[i+CW]*temp[cc][k];
         }
         costblur[r][j][k] = sum;
      }
   }
}

#define A 0.7
#define B 1.0
#define Y 0.25
float dp[4][ROWS][WDTH+1][FMAX+1]; // LO, LM, RM, RO

void reconstruct(float (&dis)[ROWS][COLS], int r) {
   int par[4][WDTH+1][FMAX+1];
   memset(par, 0, sizeof(int)*4*(WDTH+1)*(FMAX+1));
   for (int i=0; i<=WDTH; i++)
      for (int j=0; j<=FMAX; j++) { // col1 = s+i, col2 = min(s+i+j, COLS-s)
         if (i==0 || j==0) {
            for (int m=0; m<4; m++) {
               dp[m][r][i][j] = (m==3 && j==0) ? A*i:INF;
               par[m][i][j] = -1;
            }
            continue;
         }
         if (2*SZ+i+j>COLS)
            continue;
         for (int m=0; m<4; m++)
            dp[m][r][i][j] = INF;
         float cmin, min1, min2, min3, min4;
         int mx = (j+1>FMAX) ? 2:4;
         for (int m=0; m<mx; m++) {
            if (m==0) {
               min1 = dp[0][r][i][j-1]+A;
               min2 = dp[1][r][i][j-1]+B;
               min3 = dp[2][r][i][j-1]+B;
               cmin = min(min1, min(min2, min3));
               if (cmin == min1) par[m][i][j] = 0;
               if (cmin == min2) par[m][i][j] = 1;
               if (cmin == min3) par[m][i][j] = 2;
            }
            if (m==1) {
               min1 = dp[0][r][i][j-1]+B;
               min2 = dp[1][r][i][j-1]+Y;
               min3 = dp[2][r][i][j-1];
               min4 = dp[3][r][i][j-1]+B;
               cmin = min(min1, min(min2, min(min3, min4)));
               if (cmin == min1) par[m][i][j] = 0;
               if (cmin == min2) par[m][i][j] = 1;
               if (cmin == min3) par[m][i][j] = 2;
               if (cmin == min4) par[m][i][j] = 3;
               cmin+=costblur[r][i][j];
            }
            if (m==2) {
               min1 = dp[3][r][i-1][j+1]+B;
               min2 = dp[2][r][i-1][j+1]+Y;
               min3 = dp[1][r][i-1][j+1];
               min4 = dp[0][r][i-1][j+1]+B;
               cmin = min(min1, min(min2, min(min3, min4)));
               if (cmin == min1) par[m][i][j] = 13;
               if (cmin == min2) par[m][i][j] = 12;
               if (cmin == min3) par[m][i][j] = 11;
               if (cmin == min4) par[m][i][j] = 10;
               cmin+=costblur[r][i][j];
            }
            if (m==3) {
               min1 = dp[3][r][i-1][j+1]+A;
               min2 = dp[2][r][i-1][j+1]+B;
               min3 = dp[1][r][i-1][j+1]+B;
               cmin = min(min1, min(min2, min3));
               if (cmin == min1) par[m][i][j] = 13;
               if (cmin == min2) par[m][i][j] = 12;
               if (cmin == min3) par[m][i][j] = 11;
            }
            dp[m][r][i][j] = cmin;
         }
      }
      int m=3, cr = WDTH-1, cc = 1;
      while(true) {
         if (par[m][cr][cc] == -1)
            break;
         int p = par[m][cr][cc];
         if (p%10==1 || p%10==2)
            dis[r][SZ+cr] = (float)cc/FMAX;
         if (p<10) {
            cc--;
            m = p;
         }
         else {
            p%=10;
            cr--;
            cc++;
            m = p;
         }
      }
}

void dismap(float (&img_a)[ROWS][COLS][3], float (&img_b)[ROWS][COLS][3], float (&dis)[ROWS][COLS]) {
   memset(cost, 0, sizeof(float)*ROWS*(WDTH+1)*(FMAX+1));
   memset(costblur, 0, sizeof(float)*ROWS*(WDTH+1)*(FMAX+1));
   for (int i=SZ; i<ROWS-SZ; i++)
      computeCost(img_a, img_b, i);
   for (int i=SZ; i<ROWS-SZ; i++)
      blur(i);
   for (int i=SZ; i<ROWS-SZ; i++)
      reconstruct(dis, i);
}

// Proposed Algorithm
#define MAXC 10000
int K = 1000;
int dr[4] = {0, -1, 0, 1};
int dc[4] = {-1, 0, 1, 0};

int cnt[MAXC][256];
int label[ROWS][COLS];
bool vis[ROWS][COLS];

void blob(float (&dis)[ROWS][COLS], cv::Mat img) {
   // edge detection
   cv::Mat em;
   cv::blur(img, img, cv::Size(3,3));
   cv::Canny(img, em, 75, 150, 5);
   cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3), cv::Point(1, 1));
   dilate(em, em, kernel);
   //cv::imshow("EdgeMap", em);
   //cv::waitKey(0);
   float edge[ROWS][COLS];
   for (int i=0; i<ROWS; i++)
      for (int j=0; j<COLS; j++)
         edge[i][j] = (int)em.at<uchar>(i,j);
   // label components
   int cmp = 0;
   memset(cnt, 0, sizeof(int)*MAXC*256);
   memset(vis, 0, sizeof(bool)*ROWS*COLS);
   stack<int> stk;
   for (int i=SZ; i<ROWS-SZ; i++)
      for (int j=SZ; j<COLS-SZ; j++) {
         if (vis[i][j] || edge[i][j]>=128)
            continue;
         stk.push(K*i+j);
         while(stk.size()>0) {
            int idx = stk.top();
            stk.pop();
            int r = idx/K;
            int c = idx%K;
            vis[r][c] = true;
            cnt[cmp][(int)dis[r][c]]++;
            label[r][c] = cmp;
            for (int k=0; k<4; k++) {
               int y = r+dr[k];
               int x = c+dc[k];
               if (y>=0 && y<ROWS && x>=0 && x<COLS && !vis[y][x] && edge[y][x]<128)
                  stk.push(K*y+x);
            }
         }
         cmp++;
      }
   // label edges
   for (int i=0; i<ROWS; i++)
      for (int j=1; j<COLS; j++)
         if (edge[i][j]>128)
            label[i][j] = label[i][j-1];
   for (int i=0; i<ROWS; i++)
      for (int j=COLS-2; j>=0; j--)
         if (edge[i][j]>128)
            label[i][j] = label[i][j+1];
   // every component's dispairty is voted on by most pixels in region
   int mean[MAXC], lb[MAXC], ub[MAXC];
   int SW = 20;
   //cout << cmp << endl;
   for (int i=0; i<cmp; i++) {
      int psum[256];
      psum[0] = 0;
      for (int j=20; j<256; j++)
         psum[j]=cnt[i][j]+psum[j-1];
      lb[i] = SW;
      ub[i] = 255;
      if (psum[255]-psum[1] < 1600) //min component area
         continue;
      int mx = 0;
      for (int j=30; j<256-SW; j++)
         if (psum[j+SW]-psum[j] > mx) {
            mx = psum[j+SW]-psum[j];
            mean[i] = j;
         }
      int idx = mean[i];
      int avg = 0;
      for (int j=idx-SW/2; j<idx+SW/2; j++)
         avg = max(avg, cnt[i][j]);
      float F = 3.0;
      // find valleys
      while(idx>10) {
         if (cnt[i][idx]>10 && cnt[i][idx] < avg/F) {
            lb[i] = idx;
            break;
         }
         idx--;
      }
      idx = mean[i];
      while(idx<255) {
         if (cnt[i][idx]>10 && cnt[i][idx] < avg/F) {
            ub[i] = idx;
            break;
         }
         idx++;
      }
   }
   for (int i=SZ; i<ROWS-SZ; i++)
      for (int j=SZ; j<COLS-SZ; j++) {
         int cc = label[i][j];
         if (dis[i][j]<lb[cc])
            dis[i][j] = lb[cc];
         else if (dis[i][j]>ub[cc])
            dis[i][j] = ub[cc];
      }
}

string names[21] = {"Aloe", "Rocks2", "Monopoly", "Midd1", "Bowling1", "Cloth3", "Cloth4", "Baby1", "Cloth2",
                    "Lampshade1", "Wood2", "Rocks1", "Midd2", "Flowerpots", "Bowling2", "Baby2", "Baby3",
                    "Plastic", "Cloth1", "Wood1", "Lampshade2"};
void run(int cnt, string folder, string name) {
   cv::Mat li, ri;
   li = cv::imread(folder+"/"+name+"?"+to_string(cnt)+"?0.png", CV_LOAD_IMAGE_COLOR);
   ri = cv::imread(folder+"/"+name+"?"+to_string(cnt)+"?1.png", CV_LOAD_IMAGE_COLOR);
   cout << "Rows: " << li.rows << ' ' << "Cols: " << li.cols << endl;
   float left_img[ROWS][COLS][3], right_img[ROWS][COLS][3], dis[ROWS][COLS];
   memset(dis, 0, sizeof(float)*ROWS*COLS);
   toArray(left_img, li);
   toArray(right_img, ri);
   dismap(right_img, left_img, dis);
   norm(dis);
   cout << "Disparity Map Refinement Begin" << endl;
   blob(dis, ri);
   cout << "Disparity Map Refinement End" << endl;
   float final[ROWS][COLS][3];
   for (int i=0; i<ROWS; i++)
      for (int j=0; j<COLS; j++)
         for (int k=0; k<3; k++)
            final[i][j][k] = dis[i][j];
   cv::Mat ans(ROWS, COLS, CV_8UC3, cv::Scalar(0, 0, 0));
   toImage(final, ans);
   cv::imshow("Frame"+to_string(cnt), ans);
   cv::waitKey(0);
   cv::imwrite("DEMO/DIS/"+name+"?"+to_string(cnt)+".png", ans);
}

int main() {
   /*for (int s=0; s<21; s++)
      for (int c=0; c<10; c++) {
         cout << c << endl;
         run("HazyImages", c, "Lampshade2");
      }*/
   cout << "Computing Disparity Map" << endl;
   run(0, "Demo", "rocks");
}
