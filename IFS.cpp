#include <stdio.h>
#include <iostream>
#include <iterator>
#include <vector>
#include <algorithm>
#include <numeric>      // std::accumulate
#include <queue>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace cv;
#include <omp.h>

using namespace std;

//A transfer plus tard
//Code issue du cours de l'ENS pour coursera
//Mécanique statistique : algorithmes et computations
//https://www.coursera.org/course/smac
struct ligne{
  double p;
  unsigned int i1;
  unsigned int i2;
  ligne(double w, unsigned int j1, unsigned j2){
    p=w;
    i1=j1;
    i2=j2;
  }
};

class walker_table{
private :

  vector <ligne> table;
  double mean;

public:
  walker_table(vector<double> w){
    //On a just besoin d'une table de poids
    double sum = std::accumulate(w.begin(), w.end(), 0.0);
    mean = sum / double(w.size());
    
    //Vector pour le stockage prov des coef + index
    queue<pair<double,unsigned int> > long_s, short_s;
    for(unsigned int i=0; i<w.size(); i++){
      if(w[i]>mean) long_s.push(make_pair(w[i], i));
      else short_s.push(make_pair(w[i],i));
    }

    for(unsigned int i=0; i<w.size()-1; i++){
      table.push_back(ligne(short_s.front().first, short_s.front().second, 
			    long_s.front().second));
      double nw=long_s.front().first-(mean-short_s.front().first);
      if(nw<mean)
	short_s.push(make_pair(nw, long_s.front().second));
      else
	long_s.push(make_pair(nw, long_s.front().second));

      short_s.pop();
      long_s.pop();
    }
    
    if (!long_s.empty())
      table.push_back(ligne(long_s.front().first, long_s.front().second,
			    long_s.front().second));
    else
      table.push_back(ligne(short_s.front().first, short_s.front().second,
			    short_s.front().second)); 

  }

  unsigned int get_random(){
    double Upsilon=(rand() / double( RAND_MAX / (mean ) )) ;
    int i=rand()%table.size();
    if (Upsilon<table[i].p) return table[i].i1;
    else return table[i].i2;
  }

};



struct Points{
  double x;
  double y;
  Points(double xx, double yy){
    x=xx;
    y=yy;
  }
};

struct Coef{
  double a;
  double b;
  double c;
  double d;
  double e;
  double f;
  Coef(double aa, double bb, double cc,
       double dd, double ee, double ff){
    a=aa; b=bb; c=cc;
    d=dd; e=ee; f=ff;
  }
};

/* 
//Pour transformer
def trans(h,k,a,b,r,s):
    na=r*cos(a*pi/180.)
    nb=-s*sin(b*pi/180.)
    nc=h
    nd=s*sin(a*pi/180.)
    ne=r*cos(b*pi/180.)
    nf=k
    print "(%f,%f,%f, %f,%f,%f)" %((na,nb,nc,
                                    nd,ne,nf))

*/

void oper(Points &P, const Coef &C){
  double x=P.x;
  double y=P.y;
  P.x=x*C.a + y*C.b + C.c;  
  P.y=x*C.d + y*C.e + C.f;
  
}

int main( int argc, char** argv ){


  // Verification des arguments
  //1er ecriture sans arguments
  if( argc <  2 ){
    printf( "<prog> NB_iter\n" );
    return -1;
  }
  
  int it=atoi( argv[1]);

  const unsigned int dx=512;
  const unsigned int dy=512;


  Mat final=Mat::zeros(dy, dx, CV_16UC1);

  //Operateur de transformation

  vector<Coef> F;
  vector<double> poid; 
  double kx, ky, offx, offy;

  
  //tapis magique
  
  F.push_back(Coef(.5,0.,0., 0.,.5,0.));poid.push_back(200.);
  F.push_back(Coef(.5,0.,.5, 0.,.5,0.));poid.push_back(200.);
  F.push_back(Coef(.5,0.,.25, 0.,.5,.5));poid.push_back(200.);
  poid.push_back(0.);//Poids en nb paire
  kx=1.; ky=1.;
  offx=0.0; offy=0.;
  



  /*
  
  //Fougere
   //
  F.push_back(Coef(0.,0.,0., 0.,.16,0.));
  poid.push_back(1.);
  F.push_back(Coef(0.85,0.04,0., -0.04,0.85,1.600000));
  poid.push_back(85.);
  F.push_back(Coef(0.2,-0.26,0.000000, 0.23,0.22,1.600000));
  poid.push_back(07.);
  F.push_back(Coef(-0.15,0.28,0.000000, 0.26,0.24,0.440000));
  poid.push_back(07.);
  ky=0.08; kx=0.2;
  offx=2.5; //offy=2.5;
  offy=0;
  */
  
  /*
  //Branche
  F.push_back(Coef(0.387, 0.430, 0.256,   0.43, -.387, 0.522));
  poid.push_back(1.);  
  F.push_back(Coef(0.441, -0.091, 0.4219, -0.009, -.322, 0.5059));
  poid.push_back(1.);  
  F.push_back(Coef(-0.468, 0.020, 0.4, -0.113, .015, 0.4));
  poid.push_back(1.);
  poid.push_back(0.);//Poids en nb paire
  kx=1.2; ky=1.2;
  offx=0.0; offy=0.;*/


  /*
  //Crystal
  F.push_back(Coef(0.255, 0., 0.3726,   0., .255, 0.6714));
  poid.push_back(1.);
  F.push_back(Coef(0.255, 0.,0.1146,    0., .255, 0.2232));
  poid.push_back(1.);
  F.push_back(Coef(0.255, 0.,0.6306,    0., .255, 0.2232));
  poid.push_back(1.);
  F.push_back(Coef(0.370, -0.642,0.6356,   0.642, .370, -0.0061));
  poid.push_back(1.);
  kx=1.; ky=1.;
  offx=0.0; offy=0.;
  */
  ////////////////////////////////////////////////////////////////////

  walker_table tab(poid);
  Points P1(0.,0.);
  int cpt=0;


  //namedWindow( "Somme", WINDOW_NORMAL   );
  cout<< ((double) rand() / (RAND_MAX)) << "  yy  "<< ((double) rand() / (RAND_MAX))<< endl;
  
#pragma omp parallel// for num_threads(6)
  for(unsigned int i=0; i<(unsigned int) dx*dy; i++){
    

    //P1=Points(0.,0.);
    P1 = Points((double) rand() / (RAND_MAX) ,
                (double) rand() / (RAND_MAX) );


    for( int k=0; k<it; k++){
      oper(P1, F[tab.get_random()]);
    }
    unsigned int row=(P1.y+offy)*dy*ky;
    while(row<0) row+=dy;
    while(row>dy) row-=dy;
  
    unsigned int col=(P1.x+offx)*dx*kx*2;
    while(col<0) col+=dx;
    while(col> 2*dx) col-=2*dx;

#pragma omp critical
    {
      final.at<uchar>(row, col)=final.at<uchar>(row, col) +1 ;
    
      if (omp_get_thread_num() == 0) {
	double minVal, maxVal;
	minMaxLoc(final, &minVal, &maxVal); //find minimum and maximum intensities
	Mat draw, draw2;
	final.convertTo(draw, CV_8U, 255.0/(maxVal - minVal), -minVal * 255.0/(maxVal - minVal));
	applyColorMap(draw, draw2, COLORMAP_OCEAN);
	imshow( "Somme", draw2);
	waitKey(1);
	
	cpt++;
       
      }

    }
  

  }
  cout<<"x :"<<P1.x<<", y :"<<P1.y<<endl;
  cout<<" Test :"<< cpt<<endl;
  
  

  imshow( "Somme", final);
  imwrite( "test.png", final );
  waitKey(0);
  
 


  return 0;
}
                              
