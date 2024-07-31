#include<bits/stdc++.h>
using namespace std;

int sprintf(char *out, const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  int d, p;
  char *str;

  while(*fmt != '\0'){
    if(*fmt == '%'){
      fmt++;
      switch (*fmt)
      {
      case 's':
        str = va_arg(ap, char*);
        strcpy(out, str);
        out += strlen(str);
        break;
      
      case 'd':
        d = va_arg(ap, int);
        /* change int to char* */
        char temp_buf[sizeof(int)];
        p = 0;
        while (d)
        {
          int rem = d%10;
          d = d/10;
          char ch = (char)((int)'0' + rem);
          // cout << ch <<endl;
          temp_buf[p++] = ch;
        }
        // cout << temp_buf << endl;
        p--;
        while(p >= 0){
          // cout << temp_buf[p] << endl;
          *out = temp_buf[p];
          // cout << *out << endl;
          out++;
          p--;
        }
        break;

      default:
        break;
      }
      fmt++;
    }
    else{
      *out = *fmt;
      out++;
      fmt++;
    }
  }
  *out = '\0';

  va_end(ap);

  return 0;
}

int main(){
  char buf[128];
  sprintf(buf, "%s", "hello, world\n");
  cout << buf;
  sprintf(buf, "%d + %d = %d\n", 1, 1, 2);
  cout << buf;
  sprintf(buf, "%d + %d = %d\n", 2, 10, 12);
  cout << buf;
}