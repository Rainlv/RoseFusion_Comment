#include "include/common.h"


#define BLOCK_SIZE_X 32
#define BLOCK_SIZE_Y 32

using Matf31da = Eigen::Matrix<double, 3, 1, Eigen::DontAlign>;
using Matf61da = Eigen::Matrix<double, 6, 1, Eigen::DontAlign>;
using Matf31fa = Eigen::Matrix<float, 3, 1, Eigen::DontAlign>;
using Matf61fa = Eigen::Matrix<float, 6, 1, Eigen::DontAlign>;


namespace rosefusion {
    namespace internal {
        namespace cuda {

            /**
                * @brief 计算一个粒子，在一个顶点位置的TSDF值
             */
            __global__
            void particle_kernel(const PtrStepSz<short> tsdf_volume,                                        // tsdf体素块
                                 const PtrStep<float3> vertex_map_current,                                  // 当前帧的顶点图
                                 const PtrStep<float3> normal_map_current,                                  // 当前帧的法向图
                                 PtrStep<int> search_value,                                                 // 搜索范围的值
                                 PtrStep<int> search_count,                                                 // 搜索范围的计数
                                 const Eigen::Matrix<float, 3, 3, Eigen::DontAlign> rotation_current,       // 上一帧的旋转矩阵
                                 const Matf31fa translation_current,                                        // 上一帧的平移向量
                                 const Eigen::Matrix<float, 3, 3, Eigen::DontAlign> rotation_previous_inv,  // 上一帧的旋转矩阵的逆
                                 const Matf31fa translation_previous,                                       // 上一帧的平移向量
                                 const PtrStep<float> quaternion_trans,                                     // 6D预采样位姿
                                 const CameraParameters cam_params,                                         // 相机内参
                                 const int3 volume_size,                                                    // 体素块的尺寸
                                 const float voxel_scale,                                                   // 体素块的分辨率
                                 const int particle_size,                                                   // 粒子的数量
                                 const int cols,                                                            // 图像的宽度
                                 const int rows,                                                            // 图像的高度
                                 const Matf61da search_size,                                                // 6D位姿搜索范围
                                 const int level,                                                           // 金字塔的层数 32,16,8循环
                                 const int level_index                                                      //
            ) {
                // 不考虑grid，会有重复？
                const int p = blockIdx.x * blockDim.x + threadIdx.x; // 粒子索引
                const int x = (blockIdx.y * blockDim.y + threadIdx.y) * level + level_index;  // 图像列坐标
                const int y = (blockIdx.z * blockDim.z + threadIdx.z) * level + level_index;  // 图像行坐标

                if (x >= cols || y >= rows || p >= particle_size) {
                    return;
                }


                if ((normal_map_current.ptr(y)[x].x == 0 && normal_map_current.ptr(y)[x].y == 0 &&
                     normal_map_current.ptr(y)[x].z == 0)) {
                    return;
                }

                // 获取当一个顶点的坐标
                Matf31fa vertex_current;
                vertex_current.x() = vertex_map_current.ptr(y)[x].x;
                vertex_current.y() = vertex_map_current.ptr(y)[x].y;
                vertex_current.z() = vertex_map_current.ptr(y)[x].z;

                // 将当前帧的顶点坐标转换到世界坐标系下 Pw = Rwc * Pc + twc，平移在下面，初始是上一次迭代结束的位置
                Matf31fa vertex_current_global = rotation_current * vertex_current;

                // 通过`search_size`的拉伸得到采样的粒子实际的位姿
                const float t_x = quaternion_trans.ptr(p)[0] * search_size(0, 0) * 1000;
                const float t_y = quaternion_trans.ptr(p)[1] * search_size(1, 0) * 1000;
                const float t_z = quaternion_trans.ptr(p)[2] * search_size(2, 0) * 1000;

                const float q1 = quaternion_trans.ptr(p)[3] * search_size(3, 0);
                const float q2 = quaternion_trans.ptr(p)[4] * search_size(4, 0);
                const float q3 = quaternion_trans.ptr(p)[5] * search_size(5, 0);
                const float q0 = sqrt(1 - q1 * q1 - q2 * q2 - q3 * q3);

                // p'=qpq^-1，四元数旋转，即计算顶点旋转后的坐标
                // 这里是qp，下面包含与剩下的q^-1相乘
                float q_w = -(vertex_current_global.x() * q1 + vertex_current_global.y() * q2 +
                              vertex_current_global.z() * q3);
                float q_x = q0 * vertex_current_global.x() - q3 * vertex_current_global.y() +
                            q2 * vertex_current_global.z();
                float q_y = q3 * vertex_current_global.x() + q0 * vertex_current_global.y() -
                            q1 * vertex_current_global.z();
                float q_z = -q2 * vertex_current_global.x() + q1 * vertex_current_global.y() +
                            q0 * vertex_current_global.z();
                // vertex_current_global为采样点经过位姿变换后顶点的全局坐标
                vertex_current_global.x() =
                        q_x * q0 + q_w * (-q1) - q_z * (-q2) + q_y * (-q3) + t_x + translation_current.x();
                vertex_current_global.y() =
                        q_y * q0 + q_z * (-q1) + q_w * (-q2) - q_x * (-q3) + t_y + translation_current.y();
                vertex_current_global.z() =
                        q_z * q0 - q_y * (-q1) + q_x * (-q2) + q_w * (-q3) + t_z + translation_current.z();

                // 上一帧中相机坐标系下顶点的坐标
                const Matf31fa vertex_current_camera =
                        rotation_previous_inv * (vertex_current_global - translation_previous);

                // 接着将该空间点投影到上一帧的图像坐标系中，+0.5f是为了四舍五入
                Eigen::Vector2i point;
                point.x() = __float2int_rd(
                        vertex_current_camera.x() * cam_params.focal_x / vertex_current_camera.z() +
                        cam_params.principal_x + 0.5f);
                point.y() = __float2int_rd(
                        vertex_current_camera.y() * cam_params.focal_y / vertex_current_camera.z() +
                        cam_params.principal_y + 0.5f);

                // 检查投影点是否在图像中，即重叠区域
                if (point.x() >= 0 && point.y() >= 0 && point.x() < cols && point.y() < rows &&
                    vertex_current_camera.z() >= 0) {

                    Vec3fda grid = (vertex_current_global) / voxel_scale;
                    // 检查点在体素范围内
                    if (grid.x() < 1 || grid.x() >= volume_size.x - 1 || grid.y() < 1 ||
                        grid.y() >= volume_size.y - 1 ||
                        grid.z() < 1 || grid.z() >= volume_size.z - 1) {
                        return;
                    }


                    int tsdf = static_cast<int>(tsdf_volume.ptr(
                            __float2int_rd(grid(2)) * volume_size.y + __float2int_rd(grid(1)))[__float2int_rd(
                            grid(0))]);

                    // 累加同一个粒子两帧重叠区域的TSDF值，对应Eq.20括号内的分子和分母
                    atomicAdd_system(search_value + p, abs(tsdf));
                    atomicAdd_system(search_count + p, 1);


                }

            }


            /**
             * @brief 执行一次粒子估计
             */
            bool
            particle_evaluation(const VolumeData &volume,                       // 体素网格
                                const QuaternionData &quaterinons,              // 预采样粒子集合
                                SearchData &search_data,                        //
                                const Eigen::Matrix3d &rotation_current,        // 旋转矩阵3*3
                                const Matf31da &translation_current,            // 平移3*1
                                const cv::cuda::GpuMat &vertex_map_current,     // 顶点图
                                const cv::cuda::GpuMat &normal_map_current,     // 法向图
                                const Eigen::Matrix3d &rotation_previous_inv,   //
                                const Matf31da &translation_previous,           //
                                const CameraParameters &cam_params,             // 相机参数
                                const int particle_index,                       // 粒子索引 [0, 1+20, 2+40...]
                                const int particle_size,                        // 迭代时10240，3072，1024循环，表示粒子数量
                                const Matf61da &search_size,                    // 6D位姿的搜索范围
                                const int level,                                // 深度图下采样倍数：32,16,8循环
                                const int level_index,                          //
                                Eigen::Matrix<double, 7, 1> &mean_transform,    //
                                float *min_tsdf                                 //
            ) {
                std::cout.precision(17);

                const int cols = vertex_map_current.cols;
                const int rows = vertex_map_current.rows;

                dim3 block(BLOCK_SIZE_X * BLOCK_SIZE_Y, 1, 1);   // 1024, 1, 1
                dim3 grid(1, 1, 1);
                grid.x = static_cast<unsigned int>(std::ceil(
                        (float) particle_size / block.y / block.x));  // 值域为 10 | 3 | 1
                grid.y = static_cast<unsigned int>(std::ceil((float) cols / level));  // 下采样完的图像宽度
                grid.z = static_cast<unsigned int>(std::ceil((float) rows / level));  // 下采样完的图像高度

                // 初始化为全0
                search_data.gpu_search_count[particle_index / 20].setTo(0);
                search_data.gpu_search_value[particle_index / 20].setTo(0);

                // 更新`search_data`的`gpu_search_value`和`gpu_search_count`，计算粒子的tsdf误差累计值和顶点计数
                particle_kernel<<<grid, block>>>(volume.tsdf_volume,
                                                 vertex_map_current,
                                                 normal_map_current,
                                                 search_data.gpu_search_value[particle_index / 20],
                                                 search_data.gpu_search_count[particle_index / 20],
                                                 rotation_current.cast<float>(),
                                                 translation_current.cast<float>(),
                                                 rotation_previous_inv.cast<float>(),
                                                 translation_previous.cast<float>(),
                                                 quaterinons.q[particle_index],
                                                 cam_params,
                                                 volume.volume_size,
                                                 volume.voxel_scale,
                                                 particle_size,
                                                 cols,
                                                 rows,
                                                 search_size,
                                                 level,
                                                 level_index
                );
                // 对应层的`search_data`, 单行矩阵
                cv::Mat search_data_count = search_data.search_count[particle_index / 20];
                cv::Mat search_data_value = search_data.search_value[particle_index / 20];
                search_data.gpu_search_count[particle_index / 20].download(search_data_count);
                search_data.gpu_search_value[particle_index / 20].download(search_data_value);

                cudaDeviceSynchronize();  // 所有粒子计算完毕

                // 全0的粒子，即静止模型下的tsdf累加值，也即上一帧的最优位姿
                double orgin_tsdf =
                        (double) search_data_value.ptr<int>(0)[0] / (double) search_data_count.ptr<int>(0)[0];
                int orgin_count = search_data_count.ptr<int>(0)[0];

                int count_search = 0.0; // 有效迭代次数
                const int iter_rows = particle_size;  // 10240, 3072, 1024，下面迭代用

                double sum_t_x = 0.0;
                double sum_t_y = 0.0;
                double sum_t_z = 0.0;
                double sum_q_x = 0.0;
                double sum_q_y = 0.0;
                double sum_q_z = 0.0;
                double sum_q_w = 0.0;
                double sum_weight_sum = 0.0;
                double sum_mean_tsdf = 0.0;

                // 这里是为了计算权重和加权后的变换矩阵，后面做加权平均，权值就是和静止模型的tsdf累计差
                // 筛选出位姿较为准确的粒子，对这些位姿进行加权平均，权值就是该位姿下所有顶点tsdf的累计差
                for (int i = 1; i < iter_rows; ++i) {
                    // 索引永远为0，因为是单行矩阵
                    double tsdf_value =
                            (double) search_data_value.ptr<int>(i)[0] / (double) search_data_count.ptr<int>(i)[0];
                    // tsdf累计值小于静止模型，且观测次数大于静止模型的一半，即为论文中APS的条件
                    // 是不是能综合法向提升APS可靠性？
                    if (tsdf_value < orgin_tsdf && ((search_data_count.ptr<int>(i)[0]) > (orgin_count / 2.0))) {
                        // 获取PST中粒子的位姿
                        const double tx = (double) quaterinons.q_trans[particle_index].ptr<float>(i)[0];
                        const double ty = (double) quaterinons.q_trans[particle_index].ptr<float>(i)[1];
                        const double tz = (double) quaterinons.q_trans[particle_index].ptr<float>(i)[2];
                        double qx = (double) quaterinons.q_trans[particle_index].ptr<float>(i)[3];
                        double qy = (double) quaterinons.q_trans[particle_index].ptr<float>(i)[4];
                        double qz = (double) quaterinons.q_trans[particle_index].ptr<float>(i)[5];
                        // *权重计算方式
                        const double weight = orgin_tsdf - tsdf_value;

                        // 加权和，下面会求平均
                        sum_t_x += tx * weight;
                        sum_t_y += ty * weight;
                        sum_t_z += tz * weight;
                        sum_q_x += qx * weight;
                        sum_q_y += qy * weight;
                        sum_q_z += qz * weight;

                        // 这里的qx,qy,qz是PST中的值，还没有经过`search_size`缩放，需要经过缩放才得到实际在粒子估计中用的值，这里是为了计算qw才缩放，加权结果的sum_qxyz是没有经过缩放的
                        qx = qx * (double) search_size(3, 0);
                        qy = qy * (double) search_size(4, 0);
                        qz = qz * (double) search_size(5, 0);

                        // 模长固定为1，为啥？
                        const double qw = sqrt(1 - qx * qx - qy * qy - qz * qz);

                        sum_q_w += qw * weight;

                        sum_weight_sum += weight;

                        sum_mean_tsdf += weight * tsdf_value;
                        ++count_search;

                    }
                    // 有效迭代200次
                    if (count_search == 200) {
                        break;
                    }

                } // 迭代结束


                mean_transform(0, 0) = sum_t_x;
                mean_transform(1, 0) = sum_t_y;
                mean_transform(2, 0) = sum_t_z;
                mean_transform(3, 0) = sum_q_w;
                mean_transform(4, 0) = sum_q_x;
                mean_transform(5, 0) = sum_q_y;
                mean_transform(6, 0) = sum_q_z;
                const double weight_sum = sum_weight_sum;
                double mean_tsdf = sum_mean_tsdf;

                if (count_search <= 0) {

                    *min_tsdf = orgin_tsdf * DIVSHORTMAX;
                    return false;
                }

                // 加权平均
                mean_transform = mean_transform / weight_sum;
                mean_tsdf = mean_tsdf / weight_sum;

                // 乘以search_size，缩放后得到实际的位姿
                mean_transform(0, 0) = mean_transform(0, 0) * (double) search_size(0, 0);
                mean_transform(1, 0) = mean_transform(1, 0) * (double) search_size(1, 0);
                mean_transform(2, 0) = mean_transform(2, 0) * (double) search_size(2, 0);
                double qw = mean_transform(3, 0);
                double qx = mean_transform(4, 0) * search_size(3, 0);
                double qy = mean_transform(5, 0) * search_size(4, 0);
                double qz = mean_transform(6, 0) * search_size(5, 0);

                // 模长
                double lens = 1 / sqrt(qw * qw + qx * qx + qy * qy + qz * qz);

                // 单位化
                mean_transform(3, 0) = qw * lens;
                mean_transform(4, 0) = qx * lens;
                mean_transform(5, 0) = qy * lens;
                mean_transform(6, 0) = qz * lens;

                // 相当于除以32767，tsdf保存的时候乘了32767，来节约空间，这里要除回去，获得真实的tsdf值
                *min_tsdf = mean_tsdf * DIVSHORTMAX;

                return true;
            }

        }
    }
}