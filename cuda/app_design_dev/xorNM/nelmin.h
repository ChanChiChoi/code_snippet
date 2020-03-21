#ifndef NELMIN_H
#define NELMIN_H

// Nelder-Mead Minimization Algorithm ASA047
// from the Applied Statistics Algorithms available
// in STATLIB. Adapted from the C version by J. Burkhardt
// http://people.sc.fsu.edu/~jburkardt/c_src/asa047/asa047.html
// http://www.scholarpedia.org/article/Nelder-Mead_algorithm


template<typename Real>
void getmin(vector<Real>& y,Real& ymin, int& yminIndex,int nPoints){
    ymin = y[0];
    yminIndex = 0;
    for (int i = 1; i < nPoints; i++ ) {
      if ( y[i] < ymin ) {
        ymin = y[i]; 
        yminIndex = i; 
      }
    }
}


template<typename Real>
void getmax(vector<Real>& y,Real* yWorst, int& yWorstInd,int nPoints){
      *yWorst = y[0];
      yWorstInd = 0;
      for (int i = 1; i < nPoints; i++ ) {
        if ( *yWorst < y[i] ) { 
          *yWorst = y[i]; 
          yWorstInd = i; 
        }
      }
}

template<typename Real>
void calcCentroid(vector<Real>&nPoints_dim,int yWorstInd,
                  vector<Real>&centroid,int nDim, int nPoints ){


  for (int i = 0; i < nDim; i++ ) {

    Real acc = static_cast<Real>(0); 
    for (int j = 0; j < nPoints; j++ ) { 
      if(j == yWorstInd)
        continue;
      acc += nPoints_dim[i+j*nDim]; 
    }

    centroid[i] = acc / static_cast<Real>(nDim);
  }


}

template<typename Real>
void replace(vector<Real> & nPoints_dim, vector<Real>& y,
             vector<Real>& xReplace, Real& yReplace, 
             int nDim, int yWorstInd){

  for(int i=0; i<nDim; i++){
    nPoints_dim[i+yWorstInd*nDim] = xReplace[i];
  }
  y[yWorstInd] = yReplace;

}


template<typename Real>
void reflect(Real (*func)(Real*), vector<Real>& centroid, Real const& alpha,
            vector<Real>& xReflect, Real& yReflect,
            vector<Real>& nPoints_dim, int yWorstInd,
            int nDim,int *iCount
           ){

  for (int i = 0; i < nDim; i++ ) {
    // xr = c+alpha(c−xh) 
    xReflect[i] = centroid[i] + alpha * ( centroid[i] - nPoints_dim[i+yWorstInd*nDim] );
  }
  yReflect = (*func)( &xReflect[0] );
  ++*iCount;
}


template<typename Real>
void expand(Real (*func)(Real*), vector<Real>& centroid, Real const& gamma,
            vector<Real>& xExpand, Real& yExpand,
            vector<Real>& xReflect,
            int nDim,int *iCount
           ){

     for (int i = 0; i < nDim; i++ ) {
       xExpand[i] = centroid[i] + gamma * ( xReflect[i] - centroid[i] );
     }
     yExpand = (*func)( &xExpand[0] );
     ++*iCount;
}


template<typename Real>
void contractO(Real (*func)(Real*), vector<Real>& centroid, Real const& beta,
            vector<Real>& xContract, Real& yContract,
            vector<Real>& xReflect,
            int nDim,int *iCount
           ){

     for (int i = 0; i < nDim; i++ ) {
       xContract[i] = centroid[i] + beta * ( xReflect[i] - centroid[i] );
     }
     yContract = (*func)( &xContract[0] );
     ++*iCount = *iCount + 1;
}


template<typename Real>
void contractI(Real (*func)(Real*), vector<Real>& centroid, Real const& beta,
            vector<Real>& xContract, Real& yContract,
            vector<Real>& nPoints_dim,int yWorstInd,
            int nDim,int *iCount
           ){

   for (int i = 0; i < nDim; i++ ) {
     xContract[i] = centroid[i] + beta * ( nPoints_dim[i+yWorstInd*nDim] - centroid[i] );
   }
   yContract = (*func)( &xContract[0] );
   ++*iCount;
}



template<typename Real>
void shrink(Real (*func)(Real*), vector<Real>& nPoints_dim,
           int& yminIndex, Real const& delta,
           Real* xmin, vector<Real>& y,
            int nPoints,
            int nDim,int *iCount
           ){

   for (int j = 0; j < nPoints; j++ ) {
     for (int i = 0; i < nDim; i++ ) {
       nPoints_dim[i+j*nDim] = ( nPoints_dim[i+j*nDim] + nPoints_dim[i+yminIndex*nDim] ) *delta;
       xmin[i] = nPoints_dim[i+j*nDim];
     }
    y[j] = (*func)( xmin );
     ++*iCount;
   }
}



template <typename Real>
void nelmin ( Real (*func)(Real*), int nDim, Real start[], Real xmin[], 
	      Real *yWorst, Real exceptMin, Real step[], int nIterCheck, int nMaxIter, 
	      int *iCount, int *numRestart, int *ifault ){
  const Real beta = 0.5;
  const Real gamma = 2.0;
  const Real delta = 0.5;
  const Real epsilon = 0.001;
  const Real alpha = 1.0;

  int yWorstInd,yminIndex,jCount,l,nPoints;
  Real e;
  Real expectMinTotal,x,yExpand,ymin,yReflect,yContract;
  //yShrink,z;

  //  Check the input parameters.
  if ( exceptMin <= 0.0 || nDim<1 || nIterCheck<1) {
     *ifault = 1; 
     return; 
  }

  vector<Real> nPoints_dim((nDim+1)*nDim);// [nPoints, nDim]
  vector<Real> xReflect(nDim);
  vector<Real> xExpand(nDim);
  vector<Real> xContract(nDim);
  vector<Real> xShrink(nDim);
  vector<Real> centroid(nDim);
  vector<Real> y(nDim+1);

  *iCount = 0;
  *numRestart = 0;

  jCount = 0; 
  nPoints = nDim + 1;

  e = 1.0;
  expectMinTotal = exceptMin * static_cast<Real>(nDim);

  //  Initial or restarted loop.
  for ( ; ; ) {

    // put start point onto nPoints_dim last row
    for (int i = 0; i < nDim; i++ ) { 
       nPoints_dim[i+nDim*nDim] = start[i]; 
    }
    // put start's value onto y last position
    y[nDim] = (*func)( start );
    ++*iCount;
    
    // initial simplex
    // 1- keep the will change dim's value, 
    // 2- assign the changed start point to nPoints_dim
    // 3- get value of function on changed start point
    for (int j = 0; j < nDim; j++ ) {
      Real tmp = start[j];
      start[j] += step[j] * e;

      for (int i = 0; i < nDim; i++ ) { 
        nPoints_dim[i+j*nDim] = start[i];
      }
      y[j] = (*func)( start );
      start[j] = tmp;
      ++*iCount;
    }

   // initial complete==================
 
                   
    //  Find highest and lowest Y values.  yWorst = y[yWorstInd] indicates
    //  the vertex of the simplex to be replaced.
    getmin(y,ymin,yminIndex,nPoints);

    //  Inner loop.
    for ( ; ; ) {
      if ( nMaxIter <= *iCount ) 
        break; 

      // get the max value of function
      getmax(y,yWorst,yWorstInd,nPoints);

      // calc the centroid of other points, except yWorstInd point
      calcCentroid(nPoints_dim,yWorstInd,centroid,nDim,nPoints);
      // compute  Reflection
      reflect(func, centroid, alpha, xReflect,yReflect,nPoints_dim,yWorstInd,nDim,iCount);

      // Expand: If fr<fmin 
      if ( yReflect < ymin ) {
        //compute expand
        expand(func,centroid,gamma,xExpand,yExpand,xReflect,nDim,iCount );

	//  **accetp expand or reflect**.
        if ( yReflect < yExpand ) {
          replace(nPoints_dim,y, xReflect,yReflect, nDim, yWorstInd);
        } else { 
          replace(nPoints_dim,y, xExpand,yExpand, nDim, yWorstInd);
        }


      } else { 

        //here fl<= fr
        l = 0;
        for (int i = 0; i < nPoints; i++ ) {
	  if ( yReflect < y[i] ) 
            l += 1;
        }
 
        // accept reflect 	
        if ( 1 < l ) {
          replace(nPoints_dim,y,xReflect,yReflect,nDim,yWorstInd);
        }
	//  Contraction on the Y(IHI) side of the centroid.
        //inside
        else if ( l == 0 ) {
          contractI(func,centroid,beta,xContract,yContract,nPoints_dim,yWorstInd,nDim,iCount);

	  //  Contract the whole simplex.
          //ahrink transformation
          if ( y[yWorstInd] < yContract ) {

            shrink(func,nPoints_dim,yminIndex,delta,xmin,y,nPoints,nDim,iCount);

            getmin(y,ymin,yminIndex,nPoints);

            continue;
          }
	  //  accept contract outside
          else {
            replace(nPoints_dim,y,xContract,yContract,nDim,yWorstInd);
          }

        }
        //outside
        else if ( l == 1 ) {
          contractO(func,centroid,beta,xContract,yContract,xReflect,nDim,iCount);
	  // accept
          if ( yContract <= yReflect ) {
            replace(nPoints_dim,y,xContract,yContract,nDim,yWorstInd);
          }
          else {
            // ahrink
            replace(nPoints_dim,y,xReflect,yReflect,nDim,yWorstInd);
          }
        }
      }

      //  Check if YLO improved.
      if ( y[yWorstInd] < ymin ) { 
         ymin = y[yWorstInd]; 
         yminIndex = yWorstInd; 
      }

      jCount++;
      if ( jCount < nIterCheck )  
        continue; 

      //  Check to see if minimum reached.
      if ( *iCount <= nMaxIter ) {
        jCount = 0;
	
        Real acc = static_cast<Real>(0);
        for (int i = 0; i < nPoints; i++ ) { 
          acc += y[i];
        }
        x = acc / static_cast<Real>(nPoints);
	
        acc = static_cast<Real>(0);
        for (int i = 0; i < nPoints; i++ ) {
          acc += pow ( y[i] - x, 2 );
        }
	
        if ( acc <= expectMinTotal ) 
          break;
      }
    }


    //  Factorial tests to check that YNEWLO is a local minimum.
    for (int i = 0; i < nDim; i++ ) {
        xmin[i] = nPoints_dim[i+yminIndex*nDim];
    }
    *yWorst = y[yminIndex];
    
    if ( nMaxIter < *iCount ) { 
       *ifault = 2; 
       break;
    }

    *ifault = 0;

    for (int i = 0; i < nDim; i++ ) {

      e = step[i] * epsilon;
      xmin[i] = xmin[i] + e;

      Real tmp;
      tmp = (*func)( xmin );
      ++*iCount;

      if ( tmp < *yWorst ) { 
        *ifault = 2; 
        break;
      }

      xmin[i] -= 2*e;
      tmp = (*func)( xmin );
      ++*iCount;

      if ( tmp < *yWorst ) { 
        *ifault = 2; 
        break; 
      }
      xmin[i] += e;
    }
    
    if ( *ifault == 0 ) 
      break; 

    //  Restart the procedure.
    for (int i = 0; i < nDim; i++ ) { 
      start[i] = xmin[i]; 
    }
    e = epsilon;
    *numRestart = *numRestart + 1;
  }
  return;
}
#endif
