#include "stdafx.h"

namespace fs = std::filesystem;

class DRCTSuperResolution {
public:
    DRCTSuperResolution(const std::string& model_path, bool use_gpu = false);
    cv::Mat upscale(const cv::Mat& img, int tile_size = 0, int tile_overlap = 32);

private:
    cv::Mat preprocess(const cv::Mat& img);
    cv::Mat postprocess(const cv::Mat& img);
    cv::Mat inference(const cv::Mat& img);
    cv::Mat tile_inference(const cv::Mat& img, int tile_size, int tile_overlap);

    Ort::Env env;
    Ort::SessionOptions session_options;
    std::unique_ptr<Ort::Session> session;
    std::vector<const char*> input_node_names_;
    std::vector<Ort::AllocatedStringPtr> input_node_names_allocated_;
    std::vector<const char*> output_node_names_;
    std::vector<Ort::AllocatedStringPtr> output_node_names_allocated_;
    int scale_factor;
    int window_size;
    bool use_gpu;
};

DRCTSuperResolution::DRCTSuperResolution(const std::string& model_path, bool use_gpu_)
    : env(ORT_LOGGING_LEVEL_WARNING, "DRCT"), session_options(), use_gpu(use_gpu_) {
    // GPU 사용 설정 (필요한 경우)
    if (use_gpu) {
        OrtCUDAProviderOptions cuda_options;
        session_options.AppendExecutionProvider_CUDA(cuda_options);
    }

    // 모델 경로를 std::wstring으로 변환
    std::wstring widestr = std::wstring(model_path.begin(), model_path.end());
    const wchar_t* wide_model_path = widestr.c_str();

    // 세션 생성
    session = std::make_unique<Ort::Session>(env, wide_model_path, session_options);

    // 입력 및 출력 노드 이름 가져오기
    Ort::AllocatorWithDefaultOptions allocator;

    // 입력 노드 이름 가져오기
    size_t num_input_nodes = session->GetInputCount();
    input_node_names_.reserve(num_input_nodes);
    for (size_t i = 0; i < num_input_nodes; i++) {
        auto input_name_allocated = session->GetInputNameAllocated(i, allocator);
        input_node_names_allocated_.push_back(std::move(input_name_allocated));
        input_node_names_.push_back(input_node_names_allocated_.back().get());
    }

    // 출력 노드 이름 가져오기
    size_t num_output_nodes = session->GetOutputCount();
    output_node_names_.reserve(num_output_nodes);
    for (size_t i = 0; i < num_output_nodes; i++) {
        auto output_name_allocated = session->GetOutputNameAllocated(i, allocator);
        output_node_names_allocated_.push_back(std::move(output_name_allocated));
        output_node_names_.push_back(output_node_names_allocated_.back().get());
    }

    // 스케일 팩터와 윈도우 크기 설정 (모델에 따라 수정 필요)
    scale_factor = 4;  // DRCT 모델의 업스케일 팩터
    window_size = 16;  // DRCT 모델의 윈도우 크기
}

cv::Mat DRCTSuperResolution::preprocess(const cv::Mat& img) {
    cv::Mat img_float;
    img.convertTo(img_float, CV_32FC3, 1.0 / 255.0);
    cv::cvtColor(img_float, img_float, cv::COLOR_BGR2RGB);

    // 채널별로 분리하여 NCHW 형식으로 변환
    std::vector<cv::Mat> chw(img_float.channels());
    cv::split(img_float, chw);

    return img_float;
}


cv::Mat DRCTSuperResolution::postprocess(const cv::Mat& img) {
    cv::Mat img_bgr;
    cv::cvtColor(img, img_bgr, cv::COLOR_RGB2BGR);
    img_bgr.convertTo(img_bgr, CV_8U, 255.0);
    return img_bgr;
}

cv::Mat DRCTSuperResolution::inference(const cv::Mat& img) {
    // 이미지 패딩
    int h_old = img.rows;
    int w_old = img.cols;
    int h_pad = ((h_old / window_size + 1) * window_size - h_old);
    int w_pad = ((w_old / window_size + 1) * window_size - w_old);

    cv::Mat img_padded;
    cv::copyMakeBorder(img, img_padded, 0, h_pad, 0, w_pad, cv::BORDER_REFLECT);

    cv::Mat output_img;

    // 입력 텐서 생성 및 추론 실행을 별도의 스코프로 감쌉니다.
    {
        // 입력 텐서 준비
        std::array<int64_t, 4> input_shape{ 1, img_padded.channels(), img_padded.rows, img_padded.cols };
        size_t input_tensor_size = img_padded.total() * img_padded.channels();
        std::vector<float> input_tensor_values(input_tensor_size);

        // HWC에서 CHW로 변환
        {
            std::vector<cv::Mat> chw(img_padded.channels());
            cv::split(img_padded, chw);
            for (int c = 0; c < img_padded.channels(); ++c) {
                memcpy(input_tensor_values.data() + c * img_padded.rows * img_padded.cols,
                    chw[c].data, img_padded.rows * img_padded.cols * sizeof(float));
            }
            // chw 벡터의 메모리 해제
            for (auto& mat : chw) {
                mat.release();
            }
        }

        // 메모리 정보 생성
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        // 입력 텐서 생성
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            input_tensor_values.data(),
            input_tensor_size,
            input_shape.data(),
            input_shape.size());

        // 추론 실행
        auto output_tensors = session->Run(
            Ort::RunOptions{ nullptr },
            input_node_names_.data(),
            &input_tensor,
            1,
            output_node_names_.data(),
            1);

        // 출력 텐서 처리
        float* output_data = output_tensors[0].GetTensorMutableData<float>();
        auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
        int out_channels = output_shape[1];
        int out_height = output_shape[2];
        int out_width = output_shape[3];

        // 출력 이미지를 NCHW에서 HWC로 변환
        {
            std::vector<cv::Mat> output_channels(out_channels);
            size_t channel_size = out_height * out_width;

            for (int c = 0; c < out_channels; ++c) {
                output_channels[c] = cv::Mat(out_height, out_width, CV_32FC1, output_data + c * channel_size);
            }

            cv::merge(output_channels, output_img);

            // output_channels 벡터의 메모리 해제
            for (auto& mat : output_channels) {
                mat.release();
            }
        }

        // input_tensor_values 벡터의 메모리 해제
        input_tensor_values.clear();
        input_tensor_values.shrink_to_fit();

        // Ort::Value 객체들은 스코프를 벗어나면 자동으로 소멸됩니다.
        // 따라서 명시적인 해제는 필요하지 않습니다.
    }

    // 원본 크기로 자르기
    output_img = output_img(cv::Rect(0, 0, w_old * scale_factor, h_old * scale_factor)).clone();

    return output_img;
}



cv::Mat DRCTSuperResolution::tile_inference(const cv::Mat& img, int tile_size, int tile_overlap) {
    int sf = scale_factor;
    int height = img.rows;
    int width = img.cols;

    cv::Mat output_img(height * sf, width * sf, CV_32FC3, cv::Scalar(0));
    cv::Mat weight_map = cv::Mat::zeros(output_img.size(), CV_32FC3);

    int stride = tile_size - tile_overlap;
    std::vector<int> h_idx_list;
    std::vector<int> w_idx_list;

    for (int i = 0; i < height - tile_size; i += stride) {
        h_idx_list.push_back(i);
    }
    h_idx_list.push_back(std::max(0, height - tile_size));

    for (int i = 0; i < width - tile_size; i += stride) {
        w_idx_list.push_back(i);
    }
    w_idx_list.push_back(std::max(0, width - tile_size));

    for (int h_idx : h_idx_list) {
        for (int w_idx : w_idx_list) {
            cv::Rect roi(w_idx, h_idx, tile_size, tile_size);
            cv::Mat img_tile = img(roi);

            cv::Mat sr_tile = inference(img_tile);
            cv::Mat mask = cv::Mat::ones(sr_tile.size(), CV_32FC3);

            cv::Rect out_roi(w_idx * sf, h_idx * sf, sr_tile.cols, sr_tile.rows);
            output_img(out_roi) += sr_tile;
            weight_map(out_roi) += mask;
        }
    }

    output_img /= weight_map;
    return output_img;
}

cv::Mat DRCTSuperResolution::upscale(const cv::Mat& img, int tile_size, int tile_overlap) {
    cv::Mat img_float = preprocess(img);

    cv::Mat output;
    if (tile_size > 0) {
        output = tile_inference(img_float, tile_size, tile_overlap);
    }
    else {
        output = inference(img_float);
    }

    output = postprocess(output);
    return output;
}

int main(int argc, char* argv[]) {
    // 명령행 인자 처리
    std::string model_path;
    std::string input_folder = "LRbicx4";
    std::string output_folder = "results/DRCT-L";
    int scale = 4;
    int tile_size = 0; // 0이면 타일링 없이 전체 이미지로 추론
    int tile_overlap = 32;
    bool use_gpu = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--model_path" && i + 1 < argc) {
            model_path = argv[++i];
        }
        else if (arg == "--input" && i + 1 < argc) {
            input_folder = argv[++i];
        }
        else if (arg == "--output" && i + 1 < argc) {
            output_folder = argv[++i];
        }
        else if (arg == "--scale" && i + 1 < argc) {
            scale = std::stoi(argv[++i]);
        }
        else if (arg == "--tile" && i + 1 < argc) {
            tile_size = std::stoi(argv[++i]);
        }
        else if (arg == "--tile_overlap" && i + 1 < argc) {
            tile_overlap = std::stoi(argv[++i]);
        }
        else if (arg == "--use_gpu") {
            use_gpu = true;
        }
    }

    if (model_path.empty()) {
        std::cerr << "모델 경로를 지정해야 합니다. --model_path <path>" << std::endl;
        return -1;
    }

    // DRCT 모델 초기화
    DRCTSuperResolution drct(model_path, use_gpu);

    // 출력 폴더 생성
    fs::create_directories(output_folder);

    // 입력 이미지 처리
    int idx = 0;
    for (const auto& entry : fs::directory_iterator(input_folder)) {
        if (entry.is_regular_file()) {
            std::string img_path = entry.path().string();
            std::string img_name = entry.path().stem().string();
            std::cout << "처리 중: " << idx << " " << img_name << std::endl;

            // 이미지 읽기
            cv::Mat img = cv::imread(img_path, cv::IMREAD_COLOR);
            if (img.empty()) {
                std::cerr << "이미지를 읽을 수 없습니다: " << img_path << std::endl;
                continue;
            }

            // 업스케일
            cv::Mat output_img = drct.upscale(img, tile_size, tile_overlap);

            // 결과 저장
            std::string save_path = output_folder + "/" + img_name + "_DRCT-L_X4.png";
            cv::imwrite(save_path, output_img);

            idx++;
        }
    }

    return 0;
}
