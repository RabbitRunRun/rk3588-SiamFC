//
// Created by wqy on 2019/05/8.
//

#ifndef ORZ_STREAM_ENCRYPTSTREAM_H
#define ORZ_STREAM_ENCRYPTSTREAM_H

#include "orz/io/stream/filterstream.h"
#include <iostream>
#include <vector>
#include <memory>
//#include "easy_aes.h"

#define AES_BLOCKLEN 16
#include "openssl/aes.h"

namespace orz
{

    /**
     * The EncryptInputStream transformat encrypt data to plain data
     * decrypt algorithm is aes
     */
    class EncryptInputStream : public FilterInputStream {
    public:
        using self = EncryptInputStream;

        EncryptInputStream( const self & ) = delete;

        self &operator=( const self & ) = delete;

        EncryptInputStream() = delete;

        /**
         * @param in the underlying input stream.
         * @param key the decrypt key
         */
        explicit EncryptInputStream( std::shared_ptr<InputStream> in, const std::string &key );

        virtual ~EncryptInputStream();

        /**
         * @param buffer, Pointer to an array where the read characters are stored.
         * @param len, Number of characters to read.
         * @return return the number of characters read, If no char is available because
         * the end of the stream has been reached, the 0 is returned. an exception happen will return -1.
         */
        int64_t read( char *buffer, int64_t len ) override;

        /**
         * @return when the end of input stream has been reached, return true.
         */
        bool is_eof() {
            return m_eof;
        }
    private:
        /**
         * the master buffer, have been used at decrypt
         */
        uint8_t m_master[AES_BLOCKLEN];

        /**
         * the second buffer, have been used at decrypt
         */
        //uint8_t m_second[AES_BLOCKLEN];
        /**
         * the master buffer available data length
         */
        int        m_master_datalen;
        /**
         * the master buffer current reading position
         */
        int        m_master_offset;

        /**
         * the second buffer available data length
         */
        //int        m_second_datalen;

        /**
         * the aes decrypt handle
         */
        //struct AES_ctx m_ctx;

        /**
         * the underlying input stream whether reached the end
         */
        bool       m_eof;

        /**
         * the input stream decrypt key
         */
        std::string m_key;

        /**
         * aes encrypt key
         */
        AES_KEY    m_aeskey;
    };


    class EncryptOutputStream : public FilterOutputStream {
    public:
        using self = EncryptOutputStream;

        EncryptOutputStream( const self & ) = delete;

        self &operator=( const self & ) = delete;

        EncryptOutputStream() = delete;

        /**
         * @param in the underlying output stream.
         * @param key the encrypt key
         */
        explicit EncryptOutputStream( std::shared_ptr<OutputStream> out, const std::string &key );

        virtual ~EncryptOutputStream();

        /**
         * @param buffer, Pointer to an array where the write characters are stored.
         * @param len, Number of characters to write.
         * @return return the number of characters write,
         * an exception happen will return <= 0.
         */
        int64_t write( const char *buffer, int64_t len ) override;

        /**
         * write buffer data to underlying output stream.
         */
        void flush();


    private:

        /**
         * the master buffer, have been used at decrypt
         */
        uint8_t m_master[AES_BLOCKLEN];

        /**
         * the master buffer available data length
         */
        int        m_master_datalen;

        //struct AES_ctx m_ctx;
        std::string m_key;
        //char m_cskey;

        /**
         * aes encrypt key
         */
        AES_KEY    m_aeskey;
    };


}

#endif
