//
// matcaffe.cpp provides a wrapper of the caffe::Net class as well as some
// caffe::Caffe functions so that one could easily call it from matlab.
// Note that for matlab, we will simply use float as the data type.

#include <string>
#include <vector>

#include "mex.h"

#include "caffe/caffe.hpp"

#define MEX_ARGS int nlhs, mxArray **plhs, int nrhs, const mxArray **prhs

using namespace caffe;  // NOLINT(build/namespaces)

// The pointer to the internal caffe::Net instance
static shared_ptr<SGDSolver<float> > solver_;
static shared_ptr<SGDSolver<float> > gsolver_;
static shared_ptr<SGDSolver<float> > lsolver_;
static shared_ptr<Net<float> > net_;
static int init_key = -2;

// Five things to be aware of:
//   caffe uses row-major order
//   matlab uses column-major order
//   caffe uses BGR color channel order
//   matlab uses RGB color channel order
//   images need to have the data mean subtracted
//
// Data coming in from matlab needs to be in the order
//   [width, height, channels, images]
// where width is the fastest dimension.
// Here is the rough matlab for putting image data into the correct
// format:
//   % convert from uint8 to single
//   im = single(im);
//   % reshape to a fixed size (e.g., 227x227)
//   im = imresize(im, [IMAGE_DIM IMAGE_DIM], 'bilinear');
//   % permute from RGB to BGR and subtract the data mean (already in BGR)
//   im = im(:,:,[3 2 1]) - data_mean;
//   % flip width and height to make width the fastest dimension
//   im = permute(im, [2 1 3]);
//
// If you have multiple images, cat them with cat(4, ...)
//
// The actual forward function. It takes in a cell array of 4-D arrays as
// input and outputs a cell array.


static mxArray* do_forward(const mxArray* const bottom) {
  vector<Blob<float>*>& input_blobs = net_->input_blobs();
  CHECK_EQ(static_cast<unsigned int>(mxGetDimensions(bottom)[0]),
      input_blobs.size());
  for (unsigned int i = 0; i < input_blobs.size(); ++i) {
    const mxArray* const elem = mxGetCell(bottom, i);
    CHECK(mxIsSingle(elem))
        << "MatCaffe require single-precision float point data";
    CHECK_EQ(mxGetNumberOfElements(elem), input_blobs[i]->count())
        << "MatCaffe input size does not match the input size of the network";
    const float* const data_ptr =
        reinterpret_cast<const float* const>(mxGetPr(elem));
    switch (Caffe::mode()) {
    case Caffe::CPU:
      caffe_copy(input_blobs[i]->count(), data_ptr,
          input_blobs[i]->mutable_cpu_data());
      break;
    case Caffe::GPU:
      caffe_copy(input_blobs[i]->count(), data_ptr,
          input_blobs[i]->mutable_gpu_data());
      break;
    default:
      LOG(FATAL) << "Unknown Caffe mode.";
    }  // switch (Caffe::mode())
  }
  const vector<Blob<float>*>& output_blobs = net_->ForwardPrefilled();
  mxArray* mx_out = mxCreateCellMatrix(output_blobs.size(), 1);
  for (unsigned int i = 0; i < output_blobs.size(); ++i) {
    // internally data is stored as (width, height, channels, num)
    // where width is the fastest dimension
    mwSize dims[4] = {output_blobs[i]->width(), output_blobs[i]->height(),
      output_blobs[i]->channels(), output_blobs[i]->num()};
    mxArray* mx_blob =  mxCreateNumericArray(4, dims, mxSINGLE_CLASS, mxREAL);
    mxSetCell(mx_out, i, mx_blob);
    float* data_ptr = reinterpret_cast<float*>(mxGetPr(mx_blob));
    switch (Caffe::mode()) {
    case Caffe::CPU:
      caffe_copy(output_blobs[i]->count(), output_blobs[i]->cpu_data(),
          data_ptr);
      break;
    case Caffe::GPU:
      caffe_copy(output_blobs[i]->count(), output_blobs[i]->gpu_data(),
          data_ptr);
      break;
    default:
      LOG(FATAL) << "Unknown Caffe mode.";
    }  // switch (Caffe::mode())
  }

  return mx_out;
}

static mxArray* do_backward(const mxArray* const top_diff) {
  vector<Blob<float>*>& output_blobs = net_->output_blobs();
  vector<Blob<float>*>& input_blobs = net_->input_blobs();
  CHECK_EQ(static_cast<unsigned int>(mxGetDimensions(top_diff)[0]),
      output_blobs.size());
  // First, copy the output diff
  for (unsigned int i = 0; i < output_blobs.size(); ++i) {
    const mxArray* const elem = mxGetCell(top_diff, i);
    const float* const data_ptr =
        reinterpret_cast<const float* const>(mxGetPr(elem));
    switch (Caffe::mode()) {
    case Caffe::CPU:
      caffe_copy(output_blobs[i]->count(), data_ptr,
          output_blobs[i]->mutable_cpu_diff());
      break;
    case Caffe::GPU:
      caffe_copy(output_blobs[i]->count(), data_ptr,
          output_blobs[i]->mutable_gpu_diff());
      break;
    default:
      LOG(FATAL) << "Unknown Caffe mode.";
    }  // switch (Caffe::mode())
  }
  // LOG(INFO) << "Start";
  net_->Backward();
  // LOG(INFO) << "End";
  mxArray* mx_out = mxCreateCellMatrix(input_blobs.size(), 1);
  for (unsigned int i = 0; i < input_blobs.size(); ++i) {
    // internally data is stored as (width, height, channels, num)
    // where width is the fastest dimension
    mwSize dims[4] = {input_blobs[i]->width(), input_blobs[i]->height(),
      input_blobs[i]->channels(), input_blobs[i]->num()};
    mxArray* mx_blob =  mxCreateNumericArray(4, dims, mxSINGLE_CLASS, mxREAL);
    mxSetCell(mx_out, i, mx_blob);
    float* data_ptr = reinterpret_cast<float*>(mxGetPr(mx_blob));
    switch (Caffe::mode()) {
    case Caffe::CPU:
      caffe_copy(input_blobs[i]->count(), input_blobs[i]->cpu_diff(), data_ptr);
      break;
    case Caffe::GPU:
      caffe_copy(input_blobs[i]->count(), input_blobs[i]->gpu_diff(), data_ptr);
      break;
    default:
      LOG(FATAL) << "Unknown Caffe mode.";
    }  // switch (Caffe::mode())
  }

  return mx_out;
}
static mxArray* do_compute_contribution() {
  // Step 1: compute contributions for each bottom
  // LOG(INFO) << "Step 1"; 
  net_->ComputeContributions();
  // Step 2: count the number of convolution layers
  // LOG(INFO) << "Step 2"; 

  const vector<shared_ptr<Layer<float> > >& layers = net_->layers();
  const vector<string>& layer_names = net_->layer_names();
  int num_conv_layers = 0;
  for (int i=0; i<layers.size(); i++) {
    LayerParameter_LayerType layer_type = layers[i]->type();
    if (layer_type == LayerParameter_LayerType_CONVOLUTION) {
      num_conv_layers++;
    }
  }
  // Step 3: prepare output array of structures
  //LOG(INFO) << "Step 3"; 

  mxArray* mx_net_contr;
  {
    const mwSize dims[2] = {num_conv_layers, 1};
    const char* fnames[2] = {"contributions", "layer_names"};
    mx_net_contr = mxCreateStructArray(2, dims, 2, fnames);
  }
  // Step 4: compute the accumulation of contributions on each input sampls and copy them into output
  //LOG(INFO) << "Step 4";
  int index_conv_layers = 0;
  for (int i = 0; i < layers.size(); ++i) {
    LayerParameter_LayerType layer_type = layers[i]->type();
    if (layer_type != LayerParameter_LayerType_CONVOLUTION) {
      continue;
    }
    const vector<Blob<float>*>& bottom = net_->bottom_vecs(i);
    //LOG(INFO) << "After determine conv layer";
    mxArray* mx_layer_contr;
    int num = bottom[0]->num();
    int channels = bottom[0]->channels();
    mwSize dims[2] = {channels, 1};
    mx_layer_contr = mxCreateNumericArray(2, dims, mxSINGLE_CLASS, mxREAL);
    //LOG(INFO) << "SetField";
    mxSetField(mx_net_contr, index_conv_layers, "contributions", mx_layer_contr);
    mxSetField(mx_net_contr, index_conv_layers, "layer_names", mxCreateString((layer_names[i].c_str())));
    index_conv_layers++;
    //LOG(INFO) << "Read";

    float* contr_ptr = reinterpret_cast<float*>(mxGetPr(mx_layer_contr));
    for (int n = 0; n < num; n++) {
      for (int c = 0; c < channels; c++) {
        // accumulate contribution of one feature maps at all the input sampes	
	contr_ptr[c] += bottom[0]->cpu_contr()[n*channels+c];
	}
    }
  }
  //LOG(INFO) << "Return";
  return mx_net_contr;
}

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  static void do_set_select(const mxArray* input_select) {
    const vector<shared_ptr<Layer<float> > >& layers = net_->layers();
    // Step 1: check number of select cells
    const mwSize* dim = mxGetDimensions(input_select);
    int max_dim = dim[0]>dim[1]?dim[0]:dim[1];
    int min_dim = dim[0]>dim[1]?dim[1]:dim[0];
    int num_conv_layers = 0;
    // count ConvolutionLayer number
    for (int i = 0; i < layers.size(); i++) {
      LayerParameter_LayerType layer_type = layers[i]->type();
      if (layer_type == LayerParameter_LayerType_CONVOLUTION) {
	num_conv_layers++;
      }
    }

    if (max_dim!=(num_conv_layers+1) || min_dim!=1 ) {
      LOG(ERROR) << "Wrong number of select flags. Only the select flags of \
      bottom of ConvolutionLayer and top of output layer should be given";
      mexErrMsgTxt("Wrong number of select flags. Only the select flags of \
      bottom of ConvolutionLayer and top of output layer should be given");
    }

    // Step 2: copy select flags from input to an intermedia memory
    LOG(INFO) << "Step 2";
    int conv_layer_index = 0;
    vector<shared_ptr<Blob<float> > > inter_select;
    for (int i = 0; i < layers.size(); i++) {
      LayerParameter_LayerType layer_type = layers[i]->type();
      if (layer_type != LayerParameter_LayerType_CONVOLUTION) {
	continue;
      }
      LOG(INFO) << "Flag1";
      mxArray* select = mxGetCell(input_select, conv_layer_index); 
      const vector<Blob<float>*>& bottom = net_->bottom_vecs(i);
      int c = mxGetM(select);
      LOG(INFO) << "Flag2";
      if (c != bottom[0]->channels()) {
	LOG(ERROR) << "Wrong channel number. ConvLayer: " << conv_layer_index+1 << ", ch: " << bottom[0]->channels();
	mexErrMsgTxt("Wrong channel number.");
      }
      float* select_data = reinterpret_cast<float*>(mxGetPr(select));
      shared_ptr<Blob<float> > blob_ptr(new Blob<float>(1,1,1,c));
      inter_select.push_back(blob_ptr);
      LOG(INFO) << "Flag3";
      caffe_copy(c, select_data, inter_select[conv_layer_index]->mutable_cpu_data());  
      float aa = blob_ptr->asum_data();
      LOG(INFO) << "bottom channels: " << aa;
      conv_layer_index++;    
    }
    // Top of output layer
    LOG(INFO) << "Top of output layer";
    mxArray* select = mxGetCell(input_select, conv_layer_index);
    LOG(INFO) << "Get top blob"; 
    const vector<Blob<float>*>& top = net_->top_vecs(layers.size()-1);
    LOG(INFO) << "Get dim";
    int c = mxGetM(select);
    LOG(INFO) << "Check dim";
    if (c != top[0]->channels()) {
      LOG(ERROR) << "Wrong top channel number. ConvLayer: " << conv_layer_index+1 << ", ch: " << top[0]->channels();
      mexErrMsgTxt("Wrong top channel number.");
    }
    LOG(INFO) << "after check"; 
    float* select_data = reinterpret_cast<float*>(mxGetPr(select));
    shared_ptr<Blob<float> > blob_ptr(new Blob<float>(1,1,1,c));
    inter_select.push_back(blob_ptr);
    
    caffe_copy(c, select_data, inter_select[conv_layer_index]->mutable_cpu_data());

    // Step 3: set select within Net
    LOG(INFO) << "begin Set_Select";
    net_->Set_Select(inter_select);
    LOG(INFO) << "Set_Select done";

    //

  }

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
static mxArray* do_backward2(const mxArray* const top_diff) {
  vector<Blob<float>*>& output_blobs = net_->output_blobs();
  vector<Blob<float>*>& input_blobs = net_->input_blobs();
  CHECK_EQ(static_cast<unsigned int>(mxGetDimensions(top_diff)[0]),
      output_blobs.size());
  // First, copy the output diff
  for (unsigned int i = 0; i < output_blobs.size(); ++i) {
    const mxArray* const elem = mxGetCell(top_diff, i);
    const float* const data_ptr =
        reinterpret_cast<const float* const>(mxGetPr(elem));
    switch (Caffe::mode()) {
    case Caffe::CPU:
      caffe_copy(output_blobs[i]->count(), data_ptr,
          output_blobs[i]->mutable_cpu_diff());
      break;
    case Caffe::GPU:
      caffe_copy(output_blobs[i]->count(), data_ptr,
          output_blobs[i]->mutable_gpu_diff());
      break;
    default:
      LOG(FATAL) << "Unknown Caffe mode.";
    }  // switch (Caffe::mode())
  }
  net_->Backward2();
  mxArray* mx_out = mxCreateCellMatrix(input_blobs.size(), 1);
  for (unsigned int i = 0; i < input_blobs.size(); ++i) {
    // internally data is stored as (width, height, channels, num)
    // where width is the fastest dimension
    mwSize dims[4] = {input_blobs[i]->width(), input_blobs[i]->height(),
      input_blobs[i]->channels(), input_blobs[i]->num()};
    mxArray* mx_blob =  mxCreateNumericArray(4, dims, mxSINGLE_CLASS, mxREAL);
    mxSetCell(mx_out, i, mx_blob);
    float* data_ptr = reinterpret_cast<float*>(mxGetPr(mx_blob));
    switch (Caffe::mode()) {
    case Caffe::CPU:
      caffe_copy(input_blobs[i]->count(), input_blobs[i]->cpu_diff(), data_ptr);
      break;
    case Caffe::GPU:
      caffe_copy(input_blobs[i]->count(), input_blobs[i]->gpu_diff(), data_ptr);
      break;
    default:
      LOG(FATAL) << "Unknown Caffe mode.";
    }  // switch (Caffe::mode())
  }

  return mx_out;
}
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


static mxArray* do_get_weights() {
  const vector<shared_ptr<Layer<float> > >& layers = net_->layers();
  const vector<string>& layer_names = net_->layer_names();

  // Step 1: count the number of layers with weights
  int num_layers = 0;
  {
    string prev_layer_name = "";
    for (unsigned int i = 0; i < layers.size(); ++i) {
      vector<shared_ptr<Blob<float> > >& layer_blobs = layers[i]->blobs();
      if (layer_blobs.size() == 0) {
	continue;
      }
      if (layer_names[i] != prev_layer_name) {
	prev_layer_name = layer_names[i];
	num_layers++;
      }
    }
  }

  // Step 2: prepare output array of structures
  mxArray* mx_layers;
  {
    const mwSize dims[2] = {num_layers, 1};
    const char* fnames[2] = {"weights", "layer_names"};
    mx_layers = mxCreateStructArray(2, dims, 2, fnames);
  }

  // Step 3: copy weights into output
  {
    string prev_layer_name = "";
    int mx_layer_index = 0;
    for (unsigned int i = 0; i < layers.size(); ++i) {
      vector<shared_ptr<Blob<float> > >& layer_blobs = layers[i]->blobs();
      if (layer_blobs.size() == 0) {
	continue;
      }

      mxArray* mx_layer_cells = NULL;
      if (layer_names[i] != prev_layer_name) {
	prev_layer_name = layer_names[i];
	const mwSize dims[2] = {static_cast<mwSize>(layer_blobs.size()), 1};
	mx_layer_cells = mxCreateCellArray(2, dims);
	mxSetField(mx_layers, mx_layer_index, "weights", mx_layer_cells);
	mxSetField(mx_layers, mx_layer_index, "layer_names",
	    mxCreateString(layer_names[i].c_str()));
	mx_layer_index++;
      }

      for (unsigned int j = 0; j < layer_blobs.size(); ++j) {
	// internally data is stored as (width, height, channels, num)
	// where width is the fastest dimension
	mwSize dims[4] = {layer_blobs[j]->width(), layer_blobs[j]->height(),
	  layer_blobs[j]->channels(), layer_blobs[j]->num()};

	mxArray* mx_weights =
	  mxCreateNumericArray(4, dims, mxSINGLE_CLASS, mxREAL);
	mxSetCell(mx_layer_cells, j, mx_weights);
	float* weights_ptr = reinterpret_cast<float*>(mxGetPr(mx_weights));

	switch (Caffe::mode()) {
	  case Caffe::CPU:
	    caffe_copy(layer_blobs[j]->count(), layer_blobs[j]->cpu_data(),
		weights_ptr);
	    break;
	  case Caffe::GPU:
	    caffe_copy(layer_blobs[j]->count(), layer_blobs[j]->gpu_data(),
		weights_ptr);
	    break;
	  default:
	    LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
	}
      }
    }
  }

  return mx_layers;
}

static void get_weights(MEX_ARGS) {
if (nrhs != 1){
    mexErrMsgTxt("Wrong number of arguments for get_weights");
  }
  const string& solver = mxArrayToString(prhs[0]);
  if(!solver.compare("solver")) {
    net_ = solver_->net();
    LOG(INFO) << "Reshape solver input";
  }
  else if (!solver.compare("gsolver")) {
    net_ = gsolver_->net();
    LOG(INFO) << "Reshape gsolver input";
  }
  else if (!solver.compare("lsolver")) {
    net_ = lsolver_->net();
    LOG(INFO) << "Reshape lsolver input";
  }
  else {
    mexErrMsgTxt("Unknown solver type");
  }

  plhs[0] = do_get_weights();
  net_.reset();
}

static void set_mode_cpu(MEX_ARGS) {
  Caffe::set_mode(Caffe::CPU);
}

static void set_mode_gpu(MEX_ARGS) {
  Caffe::set_mode(Caffe::GPU);
}

static void set_phase_train(MEX_ARGS) {
  Caffe::set_phase(Caffe::TRAIN);
}

static void set_phase_test(MEX_ARGS) {
  Caffe::set_phase(Caffe::TEST);
}

static void set_device(MEX_ARGS) {
  if (nrhs != 1) {
    LOG(ERROR) << "Only given " << nrhs << " arguments";
    mexErrMsgTxt("Wrong number of arguments");
  }

  int device_id = static_cast<int>(mxGetScalar(prhs[0]));
  Caffe::SetDevice(device_id);
}

static void get_init_key(MEX_ARGS) {
  plhs[0] = mxCreateDoubleScalar(init_key);
}

static void init(MEX_ARGS) {
  if (nrhs != 2) {
    LOG(ERROR) << "Only given " << nrhs << " arguments";
    mexErrMsgTxt("Wrong number of arguments");
  }

  char* param_file = mxArrayToString(prhs[0]);
  char* model_file = mxArrayToString(prhs[1]);

  net_.reset(new Net<float>(string(param_file)));
  net_->CopyTrainedLayersFrom(string(model_file));

  mxFree(param_file);
  mxFree(model_file);

  init_key = random();  // NOLINT(caffe/random_fn)

  if (nlhs == 1) {
    plhs[0] = mxCreateDoubleScalar(init_key);
  }
}

static void reset(MEX_ARGS) {
  if (net_) {
    net_.reset();
    init_key = -2;
    LOG(INFO) << "Network reset, call init before use it again";
  }
}

static void forward(MEX_ARGS) {
  if (nrhs != 1) {
    LOG(ERROR) << "Only given " << nrhs << " arguments";
    mexErrMsgTxt("Wrong number of arguments");
  }
  net_ = solver_->net();
  plhs[0] = do_forward(prhs[0]);
  net_.reset();
}

static void backward(MEX_ARGS) {
  if (nrhs != 1) {
    LOG(ERROR) << "Only given " << nrhs << " arguments";
    mexErrMsgTxt("Wrong number of arguments");
  }
  net_ = solver_->net();
  plhs[0] = do_backward(prhs[0]);
  net_.reset();
}
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
static void compute_contribution(MEX_ARGS) {
  if (nrhs !=0) {
    LOG(ERROR) << "Only given " << nrhs << " arguments";
    mexErrMsgTxt("Wrong number of arguments");
  }
  plhs[0] = do_compute_contribution();
}
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
static void set_select(MEX_ARGS) {
  if (nrhs != 1) {
    LOG(ERROR) << "Only given " << nrhs << " arguments";
    mexErrMsgTxt("Wrong number of arguments");
  }
  do_set_select(prhs[0]);
}
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
static void reduce(MEX_ARGS) {
  net_->Reduce();
}


//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

static void reshape_input(MEX_ARGS) {
  if (nrhs != 2){
    mexErrMsgTxt("Wrong number of arguments for reshape");
  }

  double* dim = mxGetPr(prhs[1]);
  int id = static_cast<int>(dim[0]);
  int num = static_cast<int>(dim[1]);
  int ch = static_cast<int>(dim[2]);
  int h = static_cast<int>(dim[3]);
  int w = static_cast<int>(dim[4]);

  LOG(INFO) << "input_id = " << id;
  LOG(INFO) << "num = " << num;
  LOG(INFO) << "channels = " << ch;

  LOG(INFO) << "height = " << h;
  LOG(INFO) << "width = " << w;
  const string& solver = mxArrayToString(prhs[0]);
  if(!solver.compare("solver")) {
    net_ = solver_->net();
    LOG(INFO) << "Reshape solver input";
  }
  else if (!solver.compare("gsolver")) {
    net_ = gsolver_->net();
    LOG(INFO) << "Reshape gsolver input";
  }
  else if (!solver.compare("lsolver")) {
    net_ = lsolver_->net();
    LOG(INFO) << "Reshape lsolver input";
  }
  else {
    mexErrMsgTxt("Unknown solver type");
  }
  net_->input_blobs()[dim[id]]->Reshape(num, ch, h, w);
  net_.reset();
}
static void is_initialized(MEX_ARGS) {
  if (!net_) {
    plhs[0] = mxCreateDoubleScalar(0);
  } else {
    plhs[0] = mxCreateDoubleScalar(1);
  }
}

static void read_mean(MEX_ARGS) {
  if (nrhs != 1) {
    mexErrMsgTxt("Usage: caffe('read_mean', 'path_to_binary_mean_file'");
    return;
  }
  const string& mean_file = mxArrayToString(prhs[0]);
  Blob<float> data_mean;
  LOG(INFO) << "Loading mean file from" << mean_file;
  BlobProto blob_proto;
  bool result = ReadProtoFromBinaryFile(mean_file.c_str(), &blob_proto);
  if (!result) {
    mexErrMsgTxt("Couldn't read the file");
    return;
  }
  data_mean.FromProto(blob_proto);
  mwSize dims[4] = {data_mean.width(), data_mean.height(),
    data_mean.channels(), data_mean.num() };
  mxArray* mx_blob =  mxCreateNumericArray(4, dims, mxSINGLE_CLASS, mxREAL);
  float* data_ptr = reinterpret_cast<float*>(mxGetPr(mx_blob));
  caffe_copy(data_mean.count(), data_mean.cpu_data(), data_ptr);
  mexWarnMsgTxt("Remember that Caffe saves in [width, height, channels]"
      " format and channels are also BGR!");
  plhs[0] = mx_blob;
}
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//caffe('init_solver', 'solver.prototxt', 'trained_model.caffemodel', 'log.txt(optional)');
static void init_solver(MEX_ARGS) {
  net_.reset();
  solver_.reset();
  //Step 1: init solver
  if (nrhs <= 1 || nrhs > 3) {
    LOG(ERROR) << "Only given " << nrhs << " arguments";
    mexErrMsgTxt("caffe_mex : Wrong number of arguments");
  }

  char* solver_file = mxArrayToString(prhs[0]);

  LOG(INFO) << "Loading from " << solver_file;
  SolverParameter solver_param;
  ReadProtoFromTextFile(solver_file, &solver_param);

  solver_.reset(new SGDSolver<float>(solver_param));

  char* model_file = mxArrayToString(prhs[1]);
  if ( !string(model_file).empty() ){
    LOG(INFO) << "Recovery from " << model_file;
    solver_->net()->CopyTrainedLayersFrom(string(model_file));	
  }
  solver_->SetIter(0);
  mxFree(model_file);
  mxFree(solver_file);
  //Setp 2: assign the network pointer of solver_ to net_
  //  net_ = solver_->net();
}
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
static void presolve(MEX_ARGS) {
  solver_->PreSolve();
}
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
static void update_net(MEX_ARGS) {
  solver_->ComputeUpdateValue();
  solver_->net()->Update();
}

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
static void init_gsolver(MEX_ARGS) {
  net_.reset();
  gsolver_.reset();
  //Step 1: init solver
  if (nrhs <= 1 || nrhs > 3) {
    LOG(ERROR) << "Only given " << nrhs << " arguments";
    mexErrMsgTxt("caffe_mex : Wrong number of arguments");
  }

  char* solver_file = mxArrayToString(prhs[0]);

  LOG(INFO) << "Loading from " << solver_file;
  SolverParameter solver_param;
  ReadProtoFromTextFile(solver_file, &solver_param);

  gsolver_.reset(new SGDSolver<float>(solver_param));

  char* model_file = mxArrayToString(prhs[1]);
  if ( !string(model_file).empty() ){
    LOG(INFO) << "Recovery from " << model_file;
    gsolver_->net()->CopyTrainedLayersFrom(string(model_file));	
  }
  gsolver_->SetIter(0);
  mxFree(model_file);
  mxFree(solver_file);
}
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
static void init_lsolver(MEX_ARGS) {
  net_.reset();
  lsolver_.reset();
  //Step 1: init solver
  if (nrhs <= 1 || nrhs > 3) {
    LOG(ERROR) << "Only given " << nrhs << " arguments";
    mexErrMsgTxt("caffe_mex : Wrong number of arguments");
  }

  char* solver_file = mxArrayToString(prhs[0]);

  LOG(INFO) << "Loading from " << solver_file;
  SolverParameter solver_param;
  ReadProtoFromTextFile(solver_file, &solver_param);

  lsolver_.reset(new SGDSolver<float>(solver_param));

  char* model_file = mxArrayToString(prhs[1]);
  if ( !string(model_file).empty() ){
    LOG(INFO) << "Recovery from " << model_file;
    lsolver_->net()->CopyTrainedLayersFrom(string(model_file));	
  }
  lsolver_->SetIter(0);
  mxFree(model_file);
  mxFree(solver_file);
}

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
static void presolve_gnet(MEX_ARGS) {
  gsolver_->PreSolve();
}
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
static void presolve_lnet(MEX_ARGS) {
  lsolver_->PreSolve();
}
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
static void update_gnet(MEX_ARGS) {
  gsolver_->ComputeUpdateValue();
  gsolver_->net()->Update();
}
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
static void update_lnet(MEX_ARGS) {
  lsolver_->ComputeUpdateValue();
  lsolver_->net()->Update();
}
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
static void forward_gnet(MEX_ARGS) {
  if (nrhs != 1) {
    LOG(ERROR) << "Only given " << nrhs << " arguments";
    mexErrMsgTxt("Wrong number of arguments");
  }
  net_ = gsolver_->net();
  plhs[0] = do_forward(prhs[0]);
  net_.reset();
}
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
static void forward_lnet(MEX_ARGS) {
  if (nrhs != 1) {
    LOG(ERROR) << "Only given " << nrhs << " arguments";
    mexErrMsgTxt("Wrong number of arguments");
  }
  net_ = lsolver_->net();
  plhs[0] = do_forward(prhs[0]);
  net_.reset();
}
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
static void backward_gnet(MEX_ARGS) {
  if (nrhs != 1) {
    LOG(ERROR) << "Only given " << nrhs << " arguments";
    mexErrMsgTxt("Wrong number of arguments");
  }
  net_ = gsolver_->net();
  plhs[0] = do_backward(prhs[0]);
  net_.reset();
}
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
static void backward_lnet(MEX_ARGS) {
  if (nrhs != 1) {
    LOG(ERROR) << "Only given " << nrhs << " arguments";
    mexErrMsgTxt("Wrong number of arguments");
  }
  net_ = lsolver_->net();
  plhs[0] = do_backward(prhs[0]);
  net_.reset();
}
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
static void backward2_lnet(MEX_ARGS) {
  if (nrhs != 1) {
    LOG(ERROR) << "Only given " << nrhs << " arguments";
    mexErrMsgTxt("Wrong number of arguments");
  }
  net_ = lsolver_->net();
  plhs[0] = do_backward2(prhs[0]);
  net_.reset();
}
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
static void backward2_gnet(MEX_ARGS) {
  if (nrhs != 1) {
    LOG(ERROR) << "Only given " << nrhs << " arguments";
    mexErrMsgTxt("Wrong number of arguments");
  }
  net_ = gsolver_->net();
  plhs[0] = do_backward2(prhs[0]);
  net_.reset();
}
/** -----------------------------------------------------------------
 ** Available commands.
 **/
struct handler_registry {
  string cmd;
  void (*func)(MEX_ARGS);
};

static handler_registry handlers[] = {
  // Public API functions
  { "forward",              forward              },
  { "backward",             backward             },
 // { "init",                 init                 },
 // { "is_initialized",       is_initialized       },
  { "set_mode_cpu",         set_mode_cpu         },
  { "set_mode_gpu",         set_mode_gpu         },
  { "set_phase_train",      set_phase_train      },
  { "set_phase_test",       set_phase_test       },
  { "set_device",           set_device           },
  { "get_weights",          get_weights          },
 // { "get_init_key",         get_init_key         },
  { "reset",                reset                },
  // { "read_mean",            read_mean            },
  // { "compute_contribution", compute_contribution },
  // { "set_select",           set_select           },
  { "reduce",		    reduce               },
  { "init_solver",          init_solver          },
  { "presolve",             presolve             },
  { "update_net",           update_net           },
  { "reshape_input",        reshape_input        },
  // global/local solver
  { "init_gsolver",         init_gsolver         },
  { "init_lsolver",         init_lsolver         },
  { "presolve_gnet",        presolve_gnet        },
  { "presolve_lnet",        presolve_lnet        },
  { "update_gnet",          update_gnet          },
  { "update_lnet",          update_lnet          },
  { "forward_gnet",         forward_gnet         },
  { "forward_lnet",         forward_lnet         },
  { "backward_gnet",        backward_gnet        },
  { "backward_lnet",        backward_lnet        },
  { "backward2_lnet",       backward2_lnet       },
  { "backward2_gnet",       backward2_gnet       },
  // The end.
  { "END",                  NULL                 },
};


/** -----------------------------------------------------------------
 ** matlab entry point: caffe(api_command, arg1, arg2, ...)
 **/
void mexFunction(MEX_ARGS) {
  if (nrhs == 0) {
    LOG(ERROR) << "No API command given";
    mexErrMsgTxt("An API command is requires");
    return;
  }

  { // Handle input command
    char *cmd = mxArrayToString(prhs[0]);
    bool dispatched = false;
    // Dispatch to cmd handler
    for (int i = 0; handlers[i].func != NULL; i++) {
      if (handlers[i].cmd.compare(cmd) == 0) {
	handlers[i].func(nlhs, plhs, nrhs-1, prhs+1);
	dispatched = true;
	break;
      }
    }
    if (!dispatched) {
      LOG(ERROR) << "Unknown command `" << cmd << "'";
      mexErrMsgTxt("API command not recognized");
    }
    mxFree(cmd);
  }
}
