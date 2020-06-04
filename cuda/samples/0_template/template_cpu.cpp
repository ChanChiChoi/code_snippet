extern "C"
void computeGold(float *ref, float *idata, unsigned int const len);


void
computeGold(float *ref, float *idata, unsigned int const len){

  float const f_len = static_cast<float>(len);

  for(unsigned int i=0; i<len; i++)
    ref[i] = idata[i]*f_len;
}
