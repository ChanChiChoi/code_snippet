//calc error functor for XOR
static int const nInput = 2;
static int const nH1 = 1;
static int const nOutput = 1;
static int const nParam = 
          (nOutput+nH1)// bias
         +(nInput*nH1) // connections between input and hidden layer
         +(nH1*nOutput)// connections between hidden and output layer
         +(nInput*nOutput)//input and output
//one example length
static int const exLen = nInput+nOutput;

struct CalcError{
  
  const Real* examples;// examples data
  const Real* p;//weights
  const int nInput;// input length
  const int exLen;//one example length

  CalcError(const Real* _examples, const Real* _p,
            const int _nInput, const int _exLen)
     :examples(_examples), p(_p), nInput(_nInput), exLen(_exLen){};

  __host__ __device__ Real
  operator()(unsigned int tid){
    // get the tid example
    register Real const *in = &examples[tid*exLen];
    
    register int index = 0;
    register Real h1 = p[index++];// get hidden bias
    register Real o = p[index++];// get output bias
    
    h1 += in[0]*p[index++];//bias_h+in_1*w1_InHidden
    h1 += in[1]*p[index++];//in_2*w2_InHidden
    h1 = G(h1);//activation of hidden

    o += in[0]*p[index++];//bias_o+w1_InOut 
    o += in[1]*p[index++];//in_2*w2_InOut
    o += h1*p[index++];//ac_h*w_HiddenOut
    //o's value is prediction

    //calc the square of diffs
    o -= in[nInput];//in_nInput is gt
    return o*o;
  } 

};
