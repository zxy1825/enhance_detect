#include "postprocess.h"

using namespace std;
using namespace Eigen;

namespace front_camera_3d
{

    MonoconPost::MonoconPost(string cfg_path)
    {
        std::ifstream ifs(cfg_path, std::ios_base::binary);
        if (!ifs.good())
        {
            LOGPE << "MonoconPost load config file failed! Cfg path is " << cfg_path;
            ifs.close();
        }
        try
        {
            const auto cfg_param = toml::parse(ifs);
            const auto &net_param = toml::find(cfg_param, "MONOCON_NET_PARAM");
            const auto &input_dims_param = toml::find(cfg_param, "MONOCON_INPUT_DIMS");
            const auto &output_dims_param =
                toml::find(cfg_param, "MONOCON_OUTPUT_DIMS");
            const auto &train_dataset_param =
                toml::find(cfg_param, "TRAIN_DATASET_PARAM");

            input_dims_.clear();
            output_dims_.clear();
            LOGPI << "Begin to load input_dims";
            for (const auto &arr : input_dims_param.as_array())
            {
                input_dims_.emplace_back(toml::find<std::vector<int>>(arr, "input_dims"));
            }
            LOGPI << "Load input_dims done";
            for (const auto &arr : output_dims_param.as_array())
            {
                output_dims_.emplace_back(
                    toml::find<std::vector<int>>(arr, "output_dims"));
            }
            LOGPI << "Load output_dims done";
            const auto &imgH = toml::find(net_param, "imgH");
            const auto &imgW = toml::find(net_param, "imgW");
            const auto &mClasses = toml::find(net_param, "mClasses");
            const auto &mTopK = toml::find(net_param, "mTopK");
            const auto &mITopK = toml::find(net_param, "mITopK");
            const auto &mTestThre = toml::find(net_param, "thresh");
            string calib_path_ = toml::find<std::string>(
                cfg_param, "CLOUD_IMAGE_PROJECT", "image_lidar_calib_path");

            mImgH_ = static_cast<int>(imgH.as_integer(std::nothrow));
            mImgW_ = static_cast<int>(imgW.as_integer(std::nothrow));
            mClasses_ = static_cast<int>(mClasses.as_integer(std::nothrow));
            mTopK_ = static_cast<int>(mTopK.as_integer(std::nothrow));
            mITopK_ = static_cast<int>(mITopK.as_integer(std::nothrow));
            mTestThre_ = static_cast<float>(mTestThre.as_floating(std::nothrow));

            LOGPI << "MonoCon mTestThre_ = " << mTestThre_;
            ;

            const auto use_nms_flag =
                toml::find(cfg_param, "MONOCON_RUNTIME_CONFIG", "use_nms_flag");
            use_nms_flag_ =
                static_cast<int>(use_nms_flag.as_integer(std::nothrow));

            getCameraExternal(calib_path_);
        }
        catch (const std::exception &e)
        {
            LOGPE << "MonoconPost load param failed, error is " << e.what();
            ifs.close();
        }
        LOGPI << "MonoconPost init done.";
        ifs.close();
    }

    void MonoconPost::postIntrinsicSetter(Eigen::Matrix<float, 3, 3> P2_matrix)
    {
        mP2_ = P2_matrix;
        mTrans4x4 = Eigen::Matrix<float, 4, 4>::Identity();
        mTrans4x4.block(0, 0, 3, 3) = mP2_.block(0, 0, 3, 3);
        LOGPI << "\ninited mTrans4x4:\n"
              << mTrans4x4;
    }

    void MonoconPost::getCameraExternal(string file_name)
    {
        std::ifstream ifs(file_name, std::ios_base::binary);
        const auto cfg_file = toml::parse(ifs);
        if (ifs.good())
        {
            const auto &calib_matrix = toml::find(cfg_file, "SLTOFRONTH60");
            const auto rotation =
                toml::find<std::vector<float>>(calib_matrix, "rotation");
            const auto transform =
                toml::find<std::vector<float>>(calib_matrix, "transform");

            Eigen::Matrix4f SL2H60_ = Eigen::Matrix4f::Identity();
            SL2H60_ << rotation[0], rotation[1], rotation[2], transform[0], rotation[3],
                rotation[4], rotation[5], transform[1], rotation[6], rotation[7],
                rotation[8], transform[2], 0.0, 0.0, 0.0, 1.0;
            H602SL_ = SL2H60_.inverse();
        }
        else
        {
            LOGPE << "camera_calibrate toml file can't open !";
        }
        ifs.close();
    }

    void MonoconPost::postProcess(std::vector<float> mCArray,
                                  std::vector<float> mTopKrray)
    {
        cameraObjects_.vec_camera_object.clear();
        mCArray_ = mCArray;
        mTopKrray_ = mTopKrray;

        mIB = output_dims_[0][0];
        mIC = output_dims_[0][1];
        mIH = output_dims_[0][2];
        mIW = output_dims_[0][3];
        mIFMSize = mIH * mIW;

        mNeckDict.clear();
        mTopKDict.clear();

        mNeckDict.insert(make_pair("center_heatmap_pred", ArrySlice({0, 3})));
        mNeckDict.insert(make_pair("kpt_heatmap_pred", ArrySlice({3, 12})));
        mNeckDict.insert(make_pair("wh_pred", ArrySlice({12, 14})));
        mNeckDict.insert(make_pair("offset_pred", ArrySlice({14, 16})));
        mNeckDict.insert(make_pair("kpt_heatmap_offset_pred", ArrySlice({16, 18})));

        mNeckDict.insert(make_pair("center2kpt_offset_pred", ArrySlice({18, 36})));
        mNeckDict.insert(make_pair("dim_pred", ArrySlice({36, 39})));
        mNeckDict.insert(make_pair("depth_pred", ArrySlice({39, 41})));
        mNeckDict.insert(make_pair("alpha_cls_pred", ArrySlice({41, 53})));
        mNeckDict.insert(make_pair("alpha_offset_pred", ArrySlice({53, 65})));

        mTopKDict.insert(make_pair("topk_scores", TopKSlice({0, 1})));
        mTopKDict.insert(make_pair("topk_inds", TopKSlice({1, 2})));

        decodeCenterHeatmapPred();

        if (ret_bboxes_3d_.size() > 0)
        {
            for (int i = 0; i < ret_bboxes_3d_nms_.rows(); i++)
            {
                CameraObject camera_object;
                camera_object.object.sensors_type = SensorsType::IMAGEFront;
                ObjectType object_type;
                if (ret_labels_nms_(i) == 0)
                {
                    object_type = ObjectType::PEDESTRIAN;
                }
                else if (ret_labels_nms_(i) == 1)
                {
                    object_type = ObjectType::VEHICLE;
                }
                else if (ret_labels_nms_(i) == 2)
                {
                    object_type = ObjectType::CAR;
                }
                else
                {
                    object_type = ObjectType::UNKNOWN;
                }
                camera_object.object.object_type = object_type;
                camera_object.object.center[0] = ret_bboxes_3d_nms_(i, 0);
                camera_object.object.center[1] = ret_bboxes_3d_nms_(i, 1);
                camera_object.object.center[2] = ret_bboxes_3d_nms_(i, 2);

                camera_object.object.length = ret_bboxes_3d_nms_(i, 3);
                camera_object.object.width = ret_bboxes_3d_nms_(i, 4);
                camera_object.object.height = ret_bboxes_3d_nms_(i, 5);
                camera_object.object.theta = ret_bboxes_3d_nms_(i, 6);

                cameraObjects_.vec_camera_object.push_back(camera_object);
            }
        }

        std::vector<Detection> objs;
        if (ret_bboxes_2d_nms_.size() > 0)
        {
            for (int i = 0; i < ret_bboxes_2d_nms_.rows(); i++)
            {
                DetectObject tmp;
                ObjectType object_type;

                if (ret_labels_nms_(i) == 0)
                {
                    object_type = ObjectType::PEDESTRIAN;
                }
                else if (ret_labels_nms_(i) == 1)
                {
                    object_type = ObjectType::VEHICLE;
                }
                else if (ret_labels_nms_(i) == 2)
                {
                    object_type = ObjectType::CAR;
                }
                else
                {
                    object_type = ObjectType::UNKNOWN;
                }
                tmp.rect.x = std::min(std::max(0, int(ret_bboxes_2d_nms_(i, 0))), mImgW_);
                tmp.rect.y = std::min(std::max(0, int(ret_bboxes_2d_nms_(i, 1))), mImgH_);

                tmp.rect.width = int(ret_bboxes_2d_nms_(i, 2) - ret_bboxes_2d_nms_(i, 0));
                if (tmp.rect.width + tmp.rect.x >= mImgW_)
                {
                    tmp.rect.width = mImgW_ - tmp.rect.x;
                }

                tmp.rect.height =
                    int(ret_bboxes_2d_nms_(i, 3) - ret_bboxes_2d_nms_(i, 1));
                if (tmp.rect.height + tmp.rect.y >= mImgW_)
                {
                    tmp.rect.height = mImgH_ - tmp.rect.y;
                }

                tmp.object_type = object_type;
                tmp.object_score = float(ret_bboxes_2d_nms_(i, 4));

                cameraObjects2D_.push_back(tmp);
            }
        }
    }

    CameraObjects MonoconPost::get3dObjects() { return cameraObjects_; }

    vector<DetectObject> MonoconPost::get2dObjects() { return cameraObjects2D_; }

    MonoconPost::MonoconPost(std::string neck_featremap, std::string topk)
        : mTrans4x4(Eigen::Matrix<float, 4, 4>::Identity())
    {
    }

    void MonoconPost::printCnpyVector(string log)
    {
    }

    vF MonoconPost::ArrySlice(vector<size_t> slices)
    {
        size_t start_slice = slices[0];
        size_t end_slice = slices[1];

        assert(end_slice > start_slice
                   ? true
                   : (std::cerr << "end_slice must lager than start_slice: x != 10\n",
                      false));

        return {mCArray_.begin() + start_slice * mIFMSize,
                mCArray_.begin() + end_slice * mIFMSize};
    }

    vF MonoconPost::TopKSlice(vector<size_t> slices)
    {
        size_t start_slice = slices[0];
        size_t end_slice = slices[1];

        assert(end_slice > start_slice
                   ? true
                   : (std::cerr << "end_slice must lager than start_slice: x != 10\n",
                      false));

        return {mTopKrray_.begin() + start_slice * mITopK_,
                mTopKrray_.begin() + end_slice * mITopK_};
    }
    void MonoconPost::PrintMap()
    {
        for (auto it = mNeckDict.begin(); it != mNeckDict.end(); ++it)
        {
            string name = it->first;
            vF slice = it->second;
            cout << "| [" << name << "].shape = (1, " << slice.size() / mIFMSize << ", "
                 << mIH << ", " << mIW << ")." << endl;
        }

        for (auto it = mTopKDict.begin(); it != mTopKDict.end(); ++it)
        {
            string name = it->first;
            vF slice = it->second;
            cout << "| [" << name << "].shape = (1, " << slice.size() << ")." << endl;
        }
    }
    void MonoconPost::ParseInputTensor()
    {
        mNeckDict.clear();
        mTopKDict.clear();
        std::cout << "mNeckDict.size() = " << mNeckDict.size() << std::endl;

        mNeckDict.insert(make_pair("center_heatmap_pred", ArrySlice({0, 3})));
        mNeckDict.insert(make_pair("kpt_heatmap_pred", ArrySlice({3, 12})));
        mNeckDict.insert(make_pair("wh_pred", ArrySlice({12, 14})));
        mNeckDict.insert(make_pair("offset_pred", ArrySlice({14, 16})));
        mNeckDict.insert(make_pair("kpt_heatmap_offset_pred", ArrySlice({16, 18})));

        mNeckDict.insert(make_pair("center2kpt_offset_pred", ArrySlice({18, 36})));
        mNeckDict.insert(make_pair("dim_pred", ArrySlice({36, 39})));
        mNeckDict.insert(make_pair("depth_pred", ArrySlice({39, 41})));
        mNeckDict.insert(make_pair("alpha_cls_pred", ArrySlice({41, 53})));
        mNeckDict.insert(make_pair("alpha_offset_pred", ArrySlice({53, 65})));

        mTopKDict.insert(make_pair("topk_scores", TopKSlice({0, 1})));
        mTopKDict.insert(make_pair("topk_inds", TopKSlice({1, 2})));

        std::cout << "mNeckDict.size() = " << mNeckDict.size() << std::endl;

        PrintMap();
    }

    void MonoconPost::getLocalMaximum(vF &center_heatmap_pred, int kernel)
    {
        int pad = (kernel - 1) / 2;
    }
    void MonoconPost::decode_heatmap(vF center_heatmap_pred)
    {
        int img_h = mImgH_;
        int img_w = mImgW_;

        int batch = 1;
        int feat_h = mIH;
        int feat_w = mIW;
        int local_maximum_kernel = 3;
    }

    MatrixXf MonoconPost::transposeAndGatherFeat(vF &feature_map,
                                                 vector<int> tensor_shape,
                                                 unsigned int *ind)
    {
        MatrixXf result(mITopK_, tensor_shape[1]);
        for (size_t i = 0; i < result.rows(); i++)
        {
            for (size_t j = 0; j < result.cols(); j++)
            {
                result(i, j) = feature_map[j * mIFMSize + ind[i]];
            }
        }
        return result;
    }

    VectorXf MonoconPost::decodeAlpha(MatrixXf &alpha_cls_pred,
                                      MatrixXf &alpha_offset_pred)
    {
        MatrixXf::Index maxIndex[mITopK_];
        for (int i = 0; i < mITopK_; ++i)
            alpha_cls_pred.row(i).maxCoeff(&maxIndex[i]);
        float angle_per_class = 2 * M_PI / float(mNumAlphaBin);
        VectorXf v_alpha(mITopK_);
        for (size_t i = 0; i < mITopK_; i++)
        {
            v_alpha(i) =
                maxIndex[i] * angle_per_class + alpha_offset_pred(i, maxIndex[i]);
            v_alpha(i) = v_alpha(i) > M_PI ? v_alpha(i) - M_PI * 2 : v_alpha(i);
            v_alpha(i) = v_alpha(i) < (M_PI * -1) ? v_alpha(i) + 2 * M_PI : v_alpha(i);
        }
        return v_alpha;
    }

    VectorXf MonoconPost::calulateRotY(MatrixXf &center2d, VectorXf &v_alpha)
    {
        vF si(mITopK_, 0);
        VectorXf rot_y(mITopK_);
        for (size_t i = 0; i < mITopK_; i++)
        {
            si[i] = mP2_(0, 0);
            rot_y(i) = v_alpha(i) + atan2(center2d(i, 0) - mP2_(0, 2), si[i]);

            rot_y(i) = rot_y(i) > (M_PI) ? rot_y(i) - M_PI * 2 : rot_y(i);
            rot_y(i) = rot_y(i) < (M_PI * (-1)) ? rot_y(i) + M_PI * 2 : rot_y(i);
        }

        return rot_y;
    }

    static bool cmp(Bndbox &predBox1, Bndbox &predBox2)
    {
        return predBox1.score > predBox2.score;
    }
    void MonoconPost::CircleNMS(std::vector<Bndbox> &predBoxes,
                                std::vector<Bndbox> &nms_pred)
    {
        std::vector<int> isDroped(predBoxes.size(), 0);
        std::vector<int> keep;
        std::sort(predBoxes.begin(), predBoxes.end(), cmp);

        for (int i = 0; i < predBoxes.size(); i++)
        {
            if (isDroped[i] == 1)
            {
                continue;
            }

            keep.push_back(i);
            nms_pred.push_back(predBoxes[i]);
            for (int j = i + 1; j < predBoxes.size(); j++)
            {
                if (isDroped[j] == 1)
                {
                    continue;
                }

                float dist = std::sqrt(((predBoxes[i].x - predBoxes[j].x), 2) +
                                       pow((predBoxes[i].z - predBoxes[j].z), 2));
                float base_width = std::min(predBoxes[i].w, predBoxes[j].w);
                if (dist <= NmsThre[predBoxes[i].id] * base_width)
                {
                    isDroped[j] = 1;
                }
            }
        }
    }

    void MonoconPost::do_nms()
    {
        std::vector<Bndbox> result;
        std::map<int, std::vector<Bndbox>> boxes_classes_map;
        std::map<int, std::vector<Bndbox>> boxes_result_map;
        for (size_t idx = 0; idx < ret_bboxes_3d_.rows(); idx++)
        {
            auto Bb = Bndbox(ret_bboxes_3d_(idx, 0), ret_bboxes_3d_(idx, 1),
                             ret_bboxes_3d_(idx, 2), ret_bboxes_3d_(idx, 5),
                             ret_bboxes_3d_(idx, 3), ret_bboxes_3d_(idx, 4),
                             ret_bboxes_3d_(idx, 6), ret_labels_(idx, 0),
                             ret_bboxes_2d_(idx, 4), idx);

            if (boxes_classes_map.count(Bb.id) == 0)
                boxes_classes_map.emplace(Bb.id, std::vector<Bndbox>());
            boxes_classes_map[Bb.id].push_back(Bb);
        }

        for (auto it = boxes_classes_map.begin(); it != boxes_classes_map.end();
             it++)
        {
            boxes_result_map.emplace(it->first, std::vector<Bndbox>());
        }
        int obj_nums = 0;
        for (auto it = boxes_classes_map.begin(); it != boxes_classes_map.end();
             it++)
        {
            CircleNMS(it->second, boxes_result_map[it->first]);
            obj_nums += boxes_result_map[it->first].size();
        }

        ret_bboxes_2d_nms_.resize(obj_nums, 5);
        ret_bboxes_3d_nms_.resize(
            obj_nums,
            7);

        ret_bboxes_2d_nms_.setZero(obj_nums, 5);
        ret_bboxes_3d_nms_.setZero(obj_nums, 7);
        ret_labels_nms_.setZero(obj_nums, 1);
        size_t idx = 0;
        for (auto it = boxes_result_map.begin(); it != boxes_result_map.end(); it++)
        {
            for (auto objs_ = it->second.begin(); objs_ != it->second.end(); objs_++)
            {
                ret_bboxes_2d_nms_.block(idx, 0, 1, 5) =
                    ret_bboxes_2d_.block(objs_->matrix_idx, 0, 1, 5);
                ret_bboxes_3d_nms_.block(idx, 0, 1, 7) =
                    ret_bboxes_3d_.block(objs_->matrix_idx, 0, 1, 7);
                ret_labels_nms_(idx, 0) = ret_labels_(objs_->matrix_idx, 0);
                idx++;
            }
        }
    }
    void MonoconPost::decodeCenterHeatmapPred()
    {
        vF topk_inds = mTopKDict["topk_inds"];
        vF topk_score = mTopKDict["topk_scores"];
        vector<int> xs(mITopK_, 0);
        vector<int> ys(mITopK_, 0);
        for (size_t i = 0; i < topk_inds.size(); i++)
        {
            mstHeatMapPoint.topk_clses[i] = int(topk_inds[i]) / mIFMSize;
            mstHeatMapPoint.topk_inds[i] = int(topk_inds[i]) % mIFMSize;
            mstHeatMapPoint.topk_ys[i] = mstHeatMapPoint.topk_inds[i] / mIW;
            mstHeatMapPoint.topk_xs[i] = mstHeatMapPoint.topk_inds[i] % mIW;
            xs[i] = mstHeatMapPoint.topk_xs[i];
            ys[i] = mstHeatMapPoint.topk_ys[i];
        }

        vF wh_pred = mNeckDict["wh_pred"];
        MatrixXf wh_gathered = transposeAndGatherFeat(
            wh_pred, {1, 2, output_dims_[0][2], output_dims_[0][3]},
            mstHeatMapPoint.topk_inds);
        vF offset_pred = mNeckDict["offset_pred"];
        MatrixXf offset_gathered = transposeAndGatherFeat(
            offset_pred, {1, 2, output_dims_[0][2], output_dims_[0][3]},
            mstHeatMapPoint.topk_inds);

        for (size_t i = 0; i < topk_inds.size(); i++)
        {
            mstHeatMapPoint.topk_xs[i] += offset_gathered(i, 0);
            mstHeatMapPoint.topk_ys[i] += offset_gathered(i, 1);
        }

        for (size_t i = 0; i < mITopK_; i++)
        {
            mBoxes2d(i, 0) =
                (mstHeatMapPoint.topk_xs[i] - wh_gathered(i, 0) / 2.0) * (mImgW_ / mIW);
            mBoxes2d(i, 1) =
                (mstHeatMapPoint.topk_ys[i] - wh_gathered(i, 1) / 2.0) * (mImgH_ / mIH);
            mBoxes2d(i, 2) =
                (mstHeatMapPoint.topk_xs[i] + wh_gathered(i, 0) / 2.0) * (mImgW_ / mIW);
            mBoxes2d(i, 3) =
                (mstHeatMapPoint.topk_ys[i] + wh_gathered(i, 1) / 2.0) * (mImgH_ / mIH);
            mBoxes2d(i, 4) = topk_score[i];
        }

        vF alpha_cls_pred = mNeckDict["alpha_cls_pred"];
        MatrixXf alpha_cls_gathered = transposeAndGatherFeat(
            alpha_cls_pred, {1, 12, output_dims_[0][2], output_dims_[0][3]},
            mstHeatMapPoint.topk_inds);

        vF alpha_offset_pred = mNeckDict["alpha_offset_pred"];
        MatrixXf alpha_offset_gathered = transposeAndGatherFeat(
            alpha_offset_pred, {1, 12, output_dims_[0][2], output_dims_[0][3]},
            mstHeatMapPoint.topk_inds);

        VectorXf v_alpha = decodeAlpha(alpha_cls_gathered, alpha_offset_gathered);

        vF depth_pred = mNeckDict["depth_pred"];
        auto depth_pred_gathered = transposeAndGatherFeat(
            depth_pred, {1, 2, output_dims_[0][2], output_dims_[0][3]},
            mstHeatMapPoint.topk_inds);

        for (size_t i = 0; i < depth_pred_gathered.rows(); ++i)
        {
            float sigma = exp(-depth_pred_gathered(i, 1));
            mBoxes2d(i, 4) *= sigma;
        }

        vF center2kpt_offset_pred = mNeckDict["center2kpt_offset_pred"];
        auto center2kpt_offset_pred_gathered = transposeAndGatherFeat(
            center2kpt_offset_pred, {1, 18, output_dims_[0][2], output_dims_[0][3]},
            mstHeatMapPoint.topk_inds);
        float x_scale = mImgW_ * 1.0 / mIW;
        float y_scale = mImgH_ * 1.0 / mIH;

        MatrixXf center_offset(mITopK_, 4);
        for (size_t i = 0; i < mITopK_; i++)
        {
            center_offset(i, 0) = center2kpt_offset_pred_gathered(i, 16);
            center_offset(i, 1) = center2kpt_offset_pred_gathered(i, 17);

            center_offset(i, 0) = (center_offset(i, 0) + xs[i]) * x_scale;
            center_offset(i, 1) = (center_offset(i, 1) + ys[i]) * y_scale;

            center_offset(i, 2) = depth_pred_gathered(i, 0);
            center_offset(i, 3) = 1;
        }
        VectorXf rot_y = calulateRotY(center_offset, v_alpha);

        MatrixXf center_3d = convertPts2dTo3d(center_offset);

        vF dim_pred = mNeckDict["dim_pred"];
        auto dim_pred_gathered = transposeAndGatherFeat(
            dim_pred, {1, 3, output_dims_[0][2], output_dims_[0][3]},
            mstHeatMapPoint.topk_inds);

        Eigen::MatrixXf bboxes_3d(mITopK_, 7);
        bboxes_3d.block(0, 0, mITopK_, 3) = center_3d.block(
            0, 0, mITopK_,
            3);
        bboxes_3d.block(0, 3, mITopK_, 3) = dim_pred_gathered.block(0, 0, mITopK_, 3);
        bboxes_3d.block(0, 6, mITopK_, 1) = rot_y.block(0, 0, mITopK_, 1);

        vector<int> mask;
        for (size_t i = 0; i < mITopK_; i++)
        {
            if (mBoxes2d(i, 4) > mTestThre_)
                mask.push_back(i);
        }

        ret_bboxes_2d_.resize(mask.size(), 5);
        ret_bboxes_3d_.resize(
            mask.size(),
            7);
        ret_labels_.resize(mask.size(), 1);

        ret_bboxes_2d_.setZero(mask.size(), 5);
        ret_bboxes_3d_.setZero(mask.size(), 7);
        ret_labels_.setZero(mask.size(), 1);

        for (size_t i = 0; i < mask.size(); i++)
        {
            ret_bboxes_2d_.block(i, 0, 1, 5) = mBoxes2d.block(mask[i], 0, 1, 5);
            ret_bboxes_3d_.block(i, 0, 1, 7) = bboxes_3d.block(mask[i], 0, 1, 7);
            ret_labels_(i, 0) = mstHeatMapPoint.topk_clses[mask[i]];
        }
        if (use_nms_flag_)
            do_nms();
        else
        {
            ret_bboxes_2d_nms_ = ret_bboxes_2d_;
            ret_bboxes_3d_nms_ = ret_bboxes_3d_;
            ret_labels_nms_ = ret_labels_;
        }

        LOGPI << "After NMS:\n";
        LOGPI << "ret_bboxes_3d_nms_\n"
              << ret_bboxes_3d_nms_;
        LOGPI << "ret_bboxes_2d_nms_\n"
              << ret_bboxes_2d_nms_;
        LOGPI << "ret_labels_nms_\n"
              << ret_labels_nms_;

        camera2SL();
        return;
    }

    void MonoconPost::camera2SL()
    {
        int rows = ret_bboxes_3d_nms_.rows();
        int cols = 7;

        LOGPI << "Raw 3d objects:\n";
        LOGPI << ret_bboxes_3d_nms_;
        Eigen::MatrixXf tmp = MatrixXf::Ones(rows, 4);
        tmp.block(0, 0, rows, 3) = ret_bboxes_3d_nms_.block(0, 0, rows, 3);

        LOGPI << "tmp :\n";
        LOGPI << tmp;
        LOGPI << "tmp transpose:\n";
        LOGPI << tmp.transpose();
        LOGPI << "transform result:\n";
        LOGPI << H602SL_.block(0, 0, 4, 4) * tmp.transpose();

        ret_bboxes_3d_nms_.block(0, 0, rows, 3) =
            (H602SL_.block(0, 0, 4, 4) * tmp.transpose()).transpose();

        LOGPI << "final  result:\n";
        LOGPI << ret_bboxes_3d_nms_;
    }

    MatrixXf MonoconPost::convertPts2dTo3d(MatrixXf &center3d)
    {
        auto &unnorm_points = center3d;
        for (size_t i = 0; i < mITopK_; i++)
        {
            unnorm_points(i, 0) *= unnorm_points(i, 2);
            unnorm_points(i, 1) *= unnorm_points(i, 2);
        }
        auto inv_viewpad = mTrans4x4.inverse().transpose();

        MatrixXf mm = unnorm_points * inv_viewpad;

        return mm;
    }
}