#ifndef __NGETHOSTBYNAME_H__
#define __NGETHOSTBYNAME_H__

#include <string>
#include <vector>

#define T_A 1 //Ipv4 address
#define T_NS 2 //Nameserver
#define T_CNAME 5 // canonical name
#define T_SOA 6 /* start of authority zone */
#define T_PTR 12 /* domain name pointer */
#define T_MX 15 //Mail server


int ngethostbyname( const char *host, int query_type, int ntimeout, std::vector<std::string> &hosts );




#endif
