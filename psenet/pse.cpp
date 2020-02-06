#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <iostream>
namespace py = pybind11 ;
/*
label_map存储1,2,3,4,...不同的值,每个独立文本行用同一个值表示。
根据值求出每个文本行的坐标点
args：
    label_map：存储label值的指针
    num_label：文本行的个数
returns：
    pts：文本坐标点
 */
std::vector< std::vector<int> >find_label_coord(
    py::array_t<int32_t, py::array::c_style> label_map,
    int num_labels){

        auto pbuf_label_map = label_map.request();
        int h = pbuf_label_map.shape[0];
        int w = pbuf_label_map.shape[1];
        auto ptr_label_map = static_cast<int32_t *>(pbuf_label_map.ptr);

        std::vector<std::vector<int32_t>> pts;
        for(size_t i = 0; i < num_labels ; ++i){
            std::vector<int> pt ;
            pts.push_back(pt);
        }

        for(size_t i = 0;i < h; ++i){
            auto p_label_map = ptr_label_map + i * w ;
            for(size_t j = 0;j < w; ++j){
                int label = p_label_map[j];
                if(label > 0){
                    pts[label-1].push_back(j);
                    pts[label-1].push_back(i);
                }
            }
        }
        return pts;

    }


int ufunc_4_cpp(
    py::array_t<int32_t, py::array::c_style> S1,
    py::array_t<int32_t, py::array::c_style> S2,
    int TAG){
        auto pbuf_S1 = S1.request();
        auto pbuf_S2 = S2.request();
        int h = pbuf_S1.shape[0];
        int w = pbuf_S2.shape[1];
        auto ptr_S1 = static_cast<int32_t *>(pbuf_S1.ptr);
        auto ptr_S2 = static_cast<int32_t *>(pbuf_S2.ptr);

        for(size_t i=1 ; i<h-1 ; ++i){
            auto p_S1 = ptr_S1 + i * w ;
            auto p_S2 = ptr_S2 + i * w ;
            auto p_S2_t = ptr_S2 + (i-1) * w;
            auto p_S2_b = ptr_S2 + (i+1) * w;
            for(size_t j=1 ; j < w-1 ; ++j){
                int label = p_S1[j];
                if(label!=0){
                    if(p_S2[j-1] == TAG){
                        p_S2[j-1] = label;
                    }
                    if(p_S2[j+1] == TAG){
                        p_S2[j+1] = label;
                    }
                    if(p_S2_t[j] == TAG){
                        p_S2_t[j] = label;
                    }
                    if(p_S2_b[j] == TAG){
                        p_S2_b[j] = label;
                    }
                }
            }
        }
        return 0;
     }

PYBIND11_MODULE(pse,m){
    m.doc() = "reimplement pse use cpp";

    m.def("find_label_coord",&find_label_coord," ",py::arg("label_map"),py::arg("num_labels"));
    m.def("ufunc_4_cpp",&ufunc_4_cpp," ",py::arg("S1"),py::arg("S2"),py::arg("TAG"));
    
}


// #include <vector>
// #include <iostream>


// std::vector< std::vector<int> > pse(){

//     std::vector< std::vector<int> > pts;
//     for(size_t i = 0 ; i < 10 ;++i){
//         std::vector<int> pt ;
//         pts.push_back(pt);
//     }
    
//     for(size_t i =0 ;i < 10 ; i++){
//         for(size_t j=0 ; j < 20 ; j++){  
//                 pts[i].push_back(i);
//                 pts[i].push_back(j);
//         }
//     }
//     return pts ;
// }

// int main(){

//     std::vector< std::vector<int> > pts = pse();
//     std::cout << " " << std::endl;
//     return 0;
// }
