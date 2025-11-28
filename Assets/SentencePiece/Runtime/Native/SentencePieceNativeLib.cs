using System;
using System.Runtime.InteropServices;

namespace SentencePiece.NET
{
    public static class SentencePieceNativeLib
    {
        private const string DllName = "libSentencepiece";

        // Struct to match the native StringArray
        [StructLayout(LayoutKind.Sequential)]
        public struct StringArray
        {
            public IntPtr pieces; // char** pointer
            public int count;
        }

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr spp_create();

        [DllImport(DllName)]
        public static extern int spp_load(IntPtr processor, [MarshalAs(UnmanagedType.LPUTF8Str)] string modelFile);

        [DllImport(DllName)]
        public static extern int spp_encode(IntPtr processor, [MarshalAs(UnmanagedType.LPUTF8Str)] string input, out IntPtr output);

        [DllImport(DllName)]
        public static extern void spp_free_array(IntPtr array);

        [DllImport(DllName)]
        public static extern StringArray spp_encode_to_pieces(IntPtr handle, string text);

        [DllImport(DllName)]
        public static extern void spp_free_string_array(StringArray array);

        [DllImport(DllName)]
        public static extern void spp_free_string(IntPtr str);

        [DllImport(DllName)]
        public static extern void spp_destroy(IntPtr processor);

        [DllImport(DllName)]
        public static extern int spp_get_piece_size(IntPtr processor);

        [DllImport(DllName)]
        public static extern int spp_piece_to_id(IntPtr processor, [MarshalAs(UnmanagedType.LPUTF8Str)] string piece);

        [DllImport(DllName)]
        public static extern IntPtr spp_id_to_piece(IntPtr processor, int pieceId);

        [DllImport(DllName)]
        [return: MarshalAs(UnmanagedType.I1)]
        public static extern bool spp_is_unknown(IntPtr processor, int pieceId);

        [DllImport(DllName)]
        [return: MarshalAs(UnmanagedType.I1)]
        public static extern bool spp_is_control(IntPtr processor, int pieceId);

        [DllImport(DllName)]
        public static extern int spp_eos_id(IntPtr processor);

        [DllImport(DllName)]
        public static extern int spp_bos_id(IntPtr processor);

        [DllImport(DllName)]
        public static extern int spp_pad_id(IntPtr processor);

        [DllImport(DllName)]
        public static extern int spp_unk_id(IntPtr processor);

        [DllImport(DllName)]
        public static extern void spp_set_encode_extra_options(IntPtr processor, [MarshalAs(UnmanagedType.LPUTF8Str)] string extraOptions);
    }
}