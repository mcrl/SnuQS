#ifndef __CREG_H__
#define __CREG_H__


namespace snuqs {

class Creg {
  public:
    int* get_buf() const;
    void set_buf(int *buf);
    int get_num_bits() const;
    void set_num_bits(int num_bits);

  private:
      int *buf_;
      int num_bits_;
};

} // namespace snuqs
  
#endif // __CREG_H__
