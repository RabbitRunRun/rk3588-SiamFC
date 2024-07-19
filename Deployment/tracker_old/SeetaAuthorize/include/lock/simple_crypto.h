#ifndef SIMPLE_CRYPTO_H_
#define SIMPLE_CRYPTO_H_

#include <string>

int request_receipts(const std::string &version, const std::string &key_code, const std::string &url, const char *filename, std::string &req_orig );
std::string create_request_receipts(const std::string &version, const std::string &key_code);
//int generate_receipts( const char *filename, const char *privatekey );

int online_check(const std::string &key_code, const std::string &device_code, const std::string &online_url);
int generate_receipts_from_memory(const std::string &content, const char *outfilename);

int generate_receipts_from_file( const char *filename, const char *outfilename);

int verify_receipts( const char *filename, const char *publickey,
                     std::string &ver_orig );
std::string get_output_filename( const std::string &filename );

#endif  // SIMPLE_CRYPTO_H_
