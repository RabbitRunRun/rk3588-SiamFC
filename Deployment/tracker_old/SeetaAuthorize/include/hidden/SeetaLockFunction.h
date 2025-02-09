#pragma once

#include <string>
#include <orz/utils/random.h>
#include <orz/io/jug/jug.h>
#include <orz/utils/log.h>
#include <orz/io/stream/stream.h>

struct SeetaLock_Function
{
    int id;
    int serial_number = 0;	
    explicit SeetaLock_Function(int id) : id(id) {}
    virtual ~SeetaLock_Function() = default;
};

static inline int next_serial_number(int num)
{
    num += 0x12340000;
    num ^= 0xABCD0000;
    num += 0x00001234;
    num ^= 0x0000DCBA;
    num += 0x67898765;
    num ^= 0xBCDEDBCA;
    return num;
}

static inline bool SeetaLockSafe_call(SeetaLock_Function *function)
{
    const int serial_number = orz::Random().next();
    function->serial_number = serial_number;
    SeetaLock_call(function);
    return function->serial_number == next_serial_number(serial_number);
}

template <int FUNC_ID>
struct SeetaLock_FunctionID : public SeetaLock_Function
{
    SeetaLock_FunctionID() : SeetaLock_Function(FUNC_ID) {}
};

static const int SeetaLock_VerifyLAN_ID = 0x1001;


struct SeetaLock_VerifyLAN : public SeetaLock_FunctionID<SeetaLock_VerifyLAN_ID>
{
	SeetaLock_VerifyLAN() : SeetaLock_VerifyLAN(0) {}
	explicit SeetaLock_VerifyLAN(int keyid)
		: in({keyid})
        , out({0})
    {}
    struct
    {
        int key_code;
    } in;
    struct
    {
        int errcode;
    } out;
};

static const int SeetaLock_ErrorMessage_ID = 0x1002;
struct SeetaLock_ErrorMessage : public SeetaLock_FunctionID<SeetaLock_ErrorMessage_ID>
{
    explicit SeetaLock_ErrorMessage(int errcode)
		: in({ errcode }),
        out({ nullptr })
    {}
    struct
    {
        int errcode;
    } in;
    struct
    {
        const char *message;
    } out;
};


static const int SeetaLock_VerifyLANParallel_ID = 0x1003;
struct SeetaLock_VerifyLANParallel : public SeetaLock_FunctionID<SeetaLock_VerifyLANParallel_ID>
{
	SeetaLock_VerifyLANParallel() : SeetaLock_VerifyLANParallel(0) {  }
    explicit SeetaLock_VerifyLANParallel(int keyid)
        : in({ keyid }),
        out({0})
    {}
    struct
    {
        int key_code;
    } in;
    struct
    {
        int errcode;
    } out;
};


static const int SeetaLock_Verify_Key_Code_ID = 0x1004;
struct SeetaLock_Verify_Key_Code : public SeetaLock_FunctionID<SeetaLock_Verify_Key_Code_ID>
{
	SeetaLock_Verify_Key_Code() : SeetaLock_Verify_Key_Code(0) {  }
    explicit SeetaLock_Verify_Key_Code(const char * keycode)
        : in({ keycode }),
        out({0})
    {}
    struct
    {
        const char * key_code;
    } in;
    struct
    {
        int errcode;
    } out;
};


static const int SeetaLock_Verify_GetModelJug_ID = 0x1005;
struct SeetaLock_Verify_GetModelJug : public SeetaLock_FunctionID<SeetaLock_Verify_GetModelJug_ID>
{
	SeetaLock_Verify_GetModelJug() : SeetaLock_Verify_GetModelJug(0) {  }
    explicit SeetaLock_Verify_GetModelJug(const char * modelfile)
        : in({ modelfile }),
        out({0, orz::jug()})
    {}
    struct
    {
        const char * file;
    } in;
    struct
    {
        int errcode;
        orz::jug modeljug;
    } out;
};



static const int SeetaLock_Verify_GetModelJug_FromStream_ID = 0x1006;
struct SeetaLock_Verify_GetModelJug_FromStream : public SeetaLock_FunctionID<SeetaLock_Verify_GetModelJug_FromStream_ID>
{
	SeetaLock_Verify_GetModelJug_FromStream() : SeetaLock_Verify_GetModelJug_FromStream(0) {  }
    explicit SeetaLock_Verify_GetModelJug_FromStream(orz::InputStream* input)
        : in({ input }),
        out({0, orz::jug()})
    {}
    struct
    {
        orz::InputStream* stream;
    } in;
    struct
    {
        int errcode;
        orz::jug modeljug;
    } out;
};


orz::jug SeetaLock_GetModelJug(const char * modelfile) {
    SeetaLock_Verify_GetModelJug verify(modelfile);
    if (!SeetaLockSafe_call(&verify))
    {
        orz::Log(orz::FATAL) << "call SeetaLock_GetModelJug failed!" << orz::crash;
    }

    return verify.out.modeljug;
}

orz::jug SeetaLock_GetModelJug(orz::InputStream* in) {
    SeetaLock_Verify_GetModelJug_FromStream verify(in);
    if (!SeetaLockSafe_call(&verify))
    {
        orz::Log(orz::FATAL) << "call SeetaLock_GetModelJug failed!" << orz::crash;
    }

    return verify.out.modeljug;
}

static const int SeetaLock_Verify_Check_Instances_ID = 0x1007;
struct SeetaLock_Verify_Check_Instances : public SeetaLock_FunctionID<SeetaLock_Verify_Check_Instances_ID>
{
    explicit SeetaLock_Verify_Check_Instances() : in({NULL}),out({0})
    {
    }
    
    explicit SeetaLock_Verify_Check_Instances(const char * strname) : in({strname}),out({0})
    {
    }
    struct
    {
        const char * name;
    } in;
    struct
    {
        int errcode;
    } out;
};


static const int SeetaLock_Verify_Request_Receipts_ID = 0x1008;
struct SeetaLock_Verify_Request_Receipts: public SeetaLock_FunctionID<SeetaLock_Verify_Request_Receipts_ID>
{
    SeetaLock_Verify_Request_Receipts() : in({NULL}), out({0})
    {
    }

    explicit SeetaLock_Verify_Request_Receipts(const char * outputfile) : in({outputfile}), out({0})
    {
    }
    struct
    {
        const char * file;
    } in;
    struct
    {
        int errcode;
    } out;
};


static const int SeetaLock_Verify_Request_Receipts_Out_ID = 0x1009;
struct SeetaLock_Verify_Request_Receipts_Out: public SeetaLock_FunctionID<SeetaLock_Verify_Request_Receipts_Out_ID>
{
    SeetaLock_Verify_Request_Receipts_Out() : in({NULL, NULL}), out({0})
    {
    }

    explicit SeetaLock_Verify_Request_Receipts_Out(const char * request_receipt, const char *request_receipt_out = NULL) : in({request_receipt, request_receipt_out}), out({0})
    {
    }
    struct
    {
        const char * infile;
        const char * outfile;
    } in;
    struct
    {
        int errcode;
    } out;
};

static const int SeetaLock_Verify_Request_Receipts_Out_Ex_ID = 0x100A;
struct SeetaLock_Verify_Request_Receipts_Out_Ex: public SeetaLock_FunctionID<SeetaLock_Verify_Request_Receipts_Out_Ex_ID>
{
    SeetaLock_Verify_Request_Receipts_Out_Ex() : in({NULL}), out({0})
    {
    }

    explicit SeetaLock_Verify_Request_Receipts_Out_Ex(const char *request_receipt_out) : in({request_receipt_out}), out({0})
    {
    }
    struct
    {
        const char * outfile;
    } in;
    struct
    {
        int errcode;
    } out;
};

