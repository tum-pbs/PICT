#include "domain_structs.h"
#include "PISO_multiblock_cuda.h"
#include "grid_gen.h"

#include <type_traits>
#include <memory>

/*
bool CompareDevice(const torch::Device &device, const torch::Device &otherDevice){
	return device.type()==otherDevice.type() && device.index()==otherDevice.index();
}*/

std::string BoundaryTypeToString(const BoundaryType bt){
	switch (bt)
	{
	case BoundaryType::DIRICHLET:
		return "DIRICHLET";
	case BoundaryType::DIRICHLET_VARYING:
		return "DIRICHLET_VARYING";
	case BoundaryType::NEUMANN:
		return "NEUMANN";
	case BoundaryType::CONNECTED_GRID:
		return "CONNECTED";
	case BoundaryType::PERIODIC:
		return "PERIODIC";
	case BoundaryType::FIXED:
		return "FIXED";
	default:
		return "UNDEFINED";
	}
}

bool IsTensorEmpty(const torch::Tensor &tensor){
	return tensor.dim()==1 && tensor.size(0)==0;
}
template <typename scalar_t>
scalar_t* getTensorDataPtr(const torch::Tensor &tensor){
	return IsTensorEmpty(tensor) ? nullptr : tensor.data_ptr<scalar_t>();
}
template <typename scalar_t>
scalar_t* getOptionalTensorDataPtr(const optional<torch::Tensor> &tensor){
	return tensor.has_value() ? tensor.value().data_ptr<scalar_t>() : nullptr;
}

/* to clone optional tensors */
optional<torch::Tensor> cloneOptionalTensor(const optional<torch::Tensor> &t){
	if(t) {
		return t.value().clone();
	} else {
		return nullopt;
	}
}

std::vector<int64_t> GetTensorShape(const torch::Tensor &tensor){
	std::vector<int64_t> shape(tensor.dim(),1);
	for(index_t dim=0;dim<tensor.dim();++dim){
		shape[dim] = tensor.size(dim);
	}
	return shape;
};
bool CheckTensor(const torch::Tensor &tensor, const index_t channels, const torch::Tensor &refTensor, const std::string &name){
	CHECK_INPUT_CUDA(tensor);
	const index_t numDims = refTensor.dim();
	TORCH_CHECK(tensor.dim()==numDims, "Dimensions of " + name + " must be " + std::to_string(numDims) + ".");
	TORCH_CHECK(tensor.size(0)==1, "Batches (dim 0) are not yet supported (must be 1).");
	TORCH_CHECK(tensor.size(1)==channels, "Channels (dim 1) of " + name + " must be " + std::to_string(channels) + ".");
	for(int dim=2;dim<numDims;++dim){
		TORCH_CHECK(refTensor.size(dim)==tensor.size(dim), "Spatial dimension " + std::to_string(dim) + " fo " + name + " must match (" + std::to_string(refTensor.size(dim)) + ").");
	}
	TORCH_CHECK(tensor.dtype()==refTensor.dtype(), name + "has wrong dtype.");
	return true;
}

struct TensorInfo{
	index_t dims;
	index_t spatialDims;
	I4 size;
	index_t batchSize;
	index_t channels;
	torch::Dtype dtype;
	optional<torch::Device> device=nullopt;
};

TensorInfo getFieldInfo(torch::Tensor tensor, const bool isStaggered){
	CHECK_INPUT_CUDA(tensor);
	TensorInfo info = {.dtype=tensor.scalar_type(), .device=tensor.device()};
	
	TORCH_CHECK(3<=tensor.dim() && tensor.dim()<=5, "Fields must be 1D, 2D, or 3D in NCDHW layout.");
	info.dims = tensor.dim();
	info.spatialDims = info.dims-2;
	
	TORCH_CHECK(tensor.size(0)==1, "batches are not yet supported, batch size must be 1.");
	info.batchSize = tensor.size(0);
	info.channels = tensor.size(1);
	
	index_t dim=0;
	for(; dim<info.spatialDims; ++dim){
		info.size.a[dim] = tensor.size(info.dims - 1 - dim);
		if(isStaggered) { info.size.a[dim] -= 1; }
		TORCH_CHECK(info.size.a[dim]>2, "all spatial dimensions must be at least 3.");
	}
	for(; dim<3; ++dim){
		info.size.a[dim] = 1;
	}
	info.size.w = info.spatialDims;
	
	return info;
}

torch::Tensor CreateTensor(const index_t batches, const index_t channels, const I4 size, const index_t dims, const torch::TensorOptions options){
	std::vector<int64_t> shape;
	shape.push_back(batches);
	shape.push_back(channels);
	for(index_t i=dims-1; 0<=i; --i){
		shape.push_back(size.a[i]);
	}
	return torch::zeros(shape, options);
}
torch::Tensor CreateTensorFromRef(const index_t batches, const index_t channels, const torch::Tensor &refTensor){
	std::vector<int64_t> shape = GetTensorShape(refTensor);
	shape[0] = batches;
	shape[1] = channels;
	auto options = torch::TensorOptions().dtype(refTensor.scalar_type()).layout(torch::kStrided).device(refTensor.device().type(), refTensor.device().index());
	return torch::zeros(shape, options);
}

index_t AxisToIndex(const std::string &side){
	if(side.compare("x")==0)  return 0;
	else if(side.compare("y")==0) return 1;
	else if(side.compare("z")==0) return 2;
	return -1;
}

index_t BoundarySideToIndex(const std::string &side){
	if(side.compare("-x")==0)  return 0;
	else if(side.compare("+x")==0) return 1;
	else if(side.compare("-y")==0) return 2;
	else if(side.compare("+y")==0) return 3;
	else if(side.compare("-z")==0) return 4;
	else if(side.compare("+z")==0) return 5;
	return -1;
}

inline dim_t BoundaryIndexToDim(const dim_t index){
	//return static_cast<dim_t>((static_cast<uint32_t>(index)>>1) & 3);
	return index>>1;
}
inline dim_t BoundaryDimToIndex(const dim_t index){
	//return static_cast<dim_t>((static_cast<uint32_t>(index)<<1));
	return index<<1;
}
inline dim_t BoundaryIndexToDirection(const dim_t index){
	return index & 1;
}

std::string BoundaryIndexToString(const index_t index){
	switch (index)
	{
	case 0:
		return "-x";
	case 1:
		return "+x";
	case 2:
		return "-y";
	case 3:
		return "+y";
	case 4:
		return "-z";
	case 5:
		return "+z";
	default:
		return "?";
	}
}



CSRmatrix::CSRmatrix(torch::Tensor &a_value, torch::Tensor &a_index, torch::Tensor &a_row) : value(a_value), index(a_index), row(a_row) {
	TORCH_CHECK(value.dim()==1, "value must be 1D.");
	TORCH_CHECK(index.dim()==1, "index must be 1D.");
	TORCH_CHECK(row.dim()==1, "row must be 1D.");
	TORCH_CHECK(value.size(0)==index.size(0), "value and index must have same size.");
	TORCH_CHECK(index.dtype()==torch_kIndex, "index must have integer type.");
	//TORCH_CHECK(index.scalar_type()==torch_kIndex, "index must have integer scalar_type."); //both scalar_type() and dtype() work here
	TORCH_CHECK(row.dtype()==torch_kIndex, "row must have integer type."); 
	TORCH_CHECK(index.device()==value.device(), "all tensors must be on the same device.");
	TORCH_CHECK(row.device()==value.device(), "all tensors must be on the same device.");
}

CSRmatrix::CSRmatrix(const int32_t numValues, const int32_t numRows, const torch::Dtype valueType, const torch::Device device){
	auto valueOptions = torch::TensorOptions().dtype(valueType).layout(torch::kStrided).device(device.type(), device.index());
	auto indexOptions = torch::TensorOptions().dtype(torch_kIndex).layout(torch::kStrided).device(device.type(), device.index());

	value = torch::zeros(numValues, valueOptions);
	index = torch::zeros(numValues, indexOptions);
	row = torch::zeros(numRows+1, indexOptions);
}

void CSRmatrix::Detach(){
	value = value.detach();
}

void CSRmatrix::CreateValue(){
	value = torch::zeros_like(value);
	isTensorChanged = true;
}

void CSRmatrix::setValue(torch::Tensor &tensor){
	TORCH_CHECK(tensor.dim()==1, "value must be 1D.");
	TORCH_CHECK(value.size(0)==tensor.size(0), "value must have the correct size.");
	TORCH_CHECK(value.dtype()==tensor.dtype(), "value must have the correct type.");
	TORCH_CHECK(value.device()==tensor.device(), "all tensors must be on the same device.");
	value = tensor;
	isTensorChanged = true;
}

std::shared_ptr<CSRmatrix> CSRmatrix::Copy() const {
	torch::Tensor v = value;
	torch::Tensor i = index;
	torch::Tensor r = row;
	return std::make_shared<CSRmatrix>(v, i, r);
}
std::shared_ptr<CSRmatrix> CSRmatrix::Clone() const {
	torch::Tensor t_value = value.clone();
	torch::Tensor t_index = index.clone();
	torch::Tensor t_row = row.clone();
	
	return std::make_shared<CSRmatrix>(t_value, t_index, t_row);
}
std::shared_ptr<CSRmatrix> CSRmatrix::WithZeroValue() const {
	torch::Tensor v = torch::zeros_like(value);
	torch::Tensor i = index;
	torch::Tensor r = row;
	return std::make_shared<CSRmatrix>(v, i, r);
}

std::shared_ptr<CSRmatrix> CSRmatrix::toType(const torch::Dtype type) const {
	torch::Tensor t_value = value.toType(type);
	torch::Tensor t_index = index.clone();
	torch::Tensor t_row = row.clone();
	
	return std::make_shared<CSRmatrix>(t_value, t_index, t_row);
}

std::string CSRmatrix::ToString() const {
	std::ostringstream repr;
	repr << "CSRmatrix( size=" << getSize() << ", rows=" << getRows() << " )";
	return repr.str();
	
}

std::string Boundary::ToString() const {
	return BoundaryTypeToString(type);
}

std::shared_ptr<const Domain> Boundary::getParentDomain() const {
	if(std::shared_ptr<const Domain> parentDomain = wp_parentDomain.lock()) {
		return parentDomain;
	} else {
		TORCH_CHECK(false, "Parent Domain is expired.");
	}
}

torch::Dtype Boundary::getDtype() const {
	return getParentDomain()->getDtype();
}
torch::Device Boundary::getDevice() const {
	return getParentDomain()->getDevice();
}
index_t Boundary::getSpatialDims() const {
	return getParentDomain()->getSpatialDims();
}
bool Boundary::hasPassiveScalar() const {
	return getParentDomain()->hasPassiveScalar();
}
index_t Boundary::getPassiveScalarChannels() const {
	return getParentDomain()->getPassiveScalarChannels();
}

/** Check spatial size of NCDHW or NDHWC tensors*/
bool checkTensorSpatialSize(torch::Tensor &t, const I4 size, bool channelsFirst){
	const index_t tensorDims = t.dim();
	const index_t offset = channelsFirst ? -1 : -2;
	if(tensorDims<3) return true; //tensor has no spatial dimensions
	const index_t dims = tensorDims-2;
	for(index_t dim=0; dim<dims; ++dim){
		if(t.size(tensorDims + offset - dim)!=size.a[dim]) { return false; }
	}
	return true;
}

I4 getTensorSpatialSize(torch::Tensor &t, bool channelsFirst){
	TORCH_CHECK(t.dim()>2, "Invalid tensor for boundary shape.")
	const index_t offset = channelsFirst ? 1 : 0;
	I4 size = {.a={1,1,1,t.dim()-2}};
	for(index_t dim=0; dim<size.w; ++dim){
		size.a[dim] = t.size(size.w+offset-dim);
	}
	return size;
}

// TODO: finish implementation
// /*
FixedBoundary::FixedBoundary(optional<torch::Tensor> velocity, BoundaryConditionType velocityType,
							// torch::Tensor &pressure, BoundaryConditionType pressureType,
							 optional<torch::Tensor> passiveScalar, optional<BoundaryConditionType> passiveScalarType,
							// const index_t passiveScalarChannels,
							 optional<torch::Tensor> transform, const std::shared_ptr<const Domain> p_parentDomain)
		: Boundary(BoundaryType::FIXED, p_parentDomain) {
	
	// valid formats:
	// static: rank 2, batch + spatialDims (1-3) elements
	// varying: rank 3-5, NCDHW
	
	if(velocity && velocity.value().dim()>2){
		//setSpatialDims(velocity.value().dim()-2);
		setSizeFromTensor(velocity.value(), true);
		//setDtypeDeviceFromTensor(velocity.value());
	} else if(passiveScalar && passiveScalar.value().dim()>2){
		setSizeFromTensor(passiveScalar.value(), true);
	} else if(transform){
		setSizeFromTensor(transform.value(), false);
	} else if(velocity){
		TORCH_CHECK(velocity.value().dim()>1, "Velocity must be 1-3D and either static (shape NC) or varying (shape NCDHW).");
	} else if(passiveScalar){
		TORCH_CHECK(false, "Can't create FixedBoundary from static passive scalar alone.")
	} else {
		TORCH_CHECK(false, "Any field or 'size' is required to create FixedBoundary.");
	}
	
	if(velocity){
		setVelocity(velocity.value());
	} else {
		CreateVelocity(true);
	}
	setVelocityType(velocityType);
	
	/* Pressure boundary is fixed to 0-Neumann, but the fields are needed for correct accessing */
	// if(pressure) ...
	CreatePressure(true);
	setPressureType(BoundaryConditionType::NEUMANN);
	
	if(passiveScalar){
		setPassiveScalar(passiveScalar.value());
	} else if(p_parentDomain->hasPassiveScalar()){
		CreatePassiveScalar(true);
	}
	if(hasPassiveScalar()){
		setPassiveScalarType(passiveScalarType.value_or(m_velocityType));
	}
	
	if(transform){
		setTransform(transform.value());
	}
	
	isTensorChanged=true;
}
/*
FixedBoundary::FixedBoundary(const I4 size, BoundaryConditionType velocityType,
				const index_t passiveScalarChannels, optional<BoundaryConditionType> passiveScalarType,
				const torch::Dtype type, const torch::Device device)
		: Boundary(BoundaryType::FIXED) {
	
	setSpatialDims(size.w);
	setSize(size);
	
	CreateVelocity(true);
	setVelocityType(velocityType);
	
	if(passiveScalarChannels>0){
		CreatePassiveScalar(true);
	}
	setPassiveScalarType(passiveScalarType.value_or(m_velocityType));
	
	isTensorChanged=true;
}
*/

#ifdef DTOR_MSG
FixedBoundary::~FixedBoundary(){
	py::print("FixedBoundary dtor");
	//DetachFwd();
}
#endif

std::shared_ptr<Boundary> FixedBoundary::Copy() const {
	torch::Tensor v = m_velocity;
	//torch::Tensor p = pressure;
	optional<torch::Tensor> s = m_passiveScalar;
	optional<torch::Tensor> t = m_transform;
	std::shared_ptr<FixedBoundary> newBound = std::make_shared<FixedBoundary>(v, m_velocityType, s, nullopt, t, getParentDomain());
	if(hasPassiveScalar()){ newBound->setPassiveScalarType(m_passiveScalarTypes.value());}
	return newBound;
}

std::shared_ptr<Boundary> FixedBoundary::Clone() const {
	torch::Tensor v = m_velocity.clone();
	//torch::Tensor p = pressure;
	optional<torch::Tensor> s = cloneOptionalTensor(m_passiveScalar);
	optional<torch::Tensor> t = cloneOptionalTensor(m_transform);
	std::shared_ptr<FixedBoundary> newBound = std::make_shared<FixedBoundary>(v, m_velocityType, s, nullopt, t, getParentDomain());
	if(hasPassiveScalar()){ newBound->setPassiveScalarType(m_passiveScalarTypes.value());}
	return newBound;
}

void FixedBoundary::DetachFwd() {
	m_velocity = m_velocity.detach();
	m_pressure = m_pressure.detach();
	if(hasPassiveScalar()) { m_passiveScalar = m_passiveScalar.value().detach(); }
	if(m_transform){ m_transform = m_transform.value().detach(); }
}
void FixedBoundary::DetachGrad() {
	m_velocity_grad = m_velocity_grad.detach();
	if(m_passiveScalar_grad) { m_passiveScalar_grad = m_passiveScalar_grad.value().detach(); }
}
void FixedBoundary::Detach() {
	DetachFwd();
	DetachGrad();
}

void FixedBoundary::setVelocity(torch::Tensor &t){
	index_t spatialDims = getSpatialDims();
	CHECK_INPUT_CUDA(t);
	TORCH_CHECK(t.size(0)==1, "Batches are not yet supported (velocity).");
	TORCH_CHECK(t.size(1)==spatialDims, "Velocity channels are invalid. Velocity must be " + std::to_string(spatialDims) + "D and either static (shape NC) or varying (shape NCDHW).");
	TORCH_CHECK(t.dim()==2 || t.dim()==(2+spatialDims),
		"Velocity spatial dimensions are invalid. Velocity must be " + std::to_string(spatialDims) + "D and either static (shape NC) or varying (shape NCDHW).");
	TORCH_CHECK(t.scalar_type()==getDtype(), "Velocity dtype must match domain.");
	bool velocityStatic = t.dim()==2;
	if(!velocityStatic && hasSize()){
		TORCH_CHECK(checkTensorSpatialSize(t, getSizes(), true), "New velocity spatial dimensions must match existing fields.");
	}
	
	m_velocity = t;
	m_velocityStatic = velocityStatic;
	
	// keep gradient coherent
	if(!IsTensorEmpty(m_velocity_grad)){
		if(!(m_velocity_grad.dim()==t.dim())){
			// TODO if grad is static: broadcast to varying, else: sum to static
			TORCH_CHECK(false, "New velocity does not match existing gradient tensor.");
		}
	}
	
	isTensorChanged=true;
}

void FixedBoundary::setVelocityType(const BoundaryConditionType velocityType){
	TORCH_CHECK(velocityType==BoundaryConditionType::DIRICHLET, "Invalid velocity boundary type: Currently only Dirichlet boundaries are supported");
	m_velocityType = velocityType;
}

void FixedBoundary::CreateVelocity(const bool createStatic){
	torch::TensorOptions valueOptions = torch::TensorOptions().dtype(getDtype()).layout(torch::kStrided).device(getDevice().type(), getDevice().index());
	torch::Tensor velocity;
	if(createStatic){
		velocity = torch::zeros({1, getSpatialDims()}, valueOptions);
	} else {
		TORCH_CHECK(hasSize(), "FixedBoundary is missing size to create a varying velocity.");
		const I4 size = getSizes();
		velocity = CreateTensor(1, getSpatialDims(), size, getSpatialDims(), valueOptions);
	}
	setVelocity(velocity);
}

#ifdef WITH_GRAD
void FixedBoundary::setVelocityGrad(torch::Tensor &t){
	CHECK_INPUT_CUDA(t);
	index_t spatialDims = getSpatialDims();
	CheckTensor(t, spatialDims, m_velocity, "FixedBoundary.velocity_grad");
	
	m_velocity_grad = t;
	isTensorChanged=true;
}

void FixedBoundary::CreateVelocityGrad(){
	torch::Tensor velocity_grad = torch::zeros_like(m_velocity);
	setVelocityGrad(velocity_grad);
}

#endif //WITH_GRAD

torch::Tensor FixedBoundary::getVelocityVarying(optional<I4> sizeopt) const {
	if(m_velocityStatic){
		TORCH_CHECK(hasSize() || sizeopt, "FixedBoundary has no fields with spatial dimensions to infer varying velocity shape.");
		torch::Tensor vel = m_velocity;
		const I4 size = hasSize() ? getSizes() : sizeopt.value();
		const index_t spatialDims = getSpatialDims();
		std::vector<int64_t> tileMul;
		// NC -> NCDHW
		for(index_t dim=0; dim<spatialDims; ++dim){
			vel = torch::unsqueeze(vel, -1);
			tileMul.push_back(size.a[spatialDims-1-dim]); // z,y,x
		}
		
		vel = torch::tile(vel, tileMul);
		
		return vel.clone().contiguous();
	} else {
		return m_velocity;
	}
}
void FixedBoundary::makeVelocityVarying(optional<I4> size){
	if(m_velocityStatic){
		torch::Tensor velVarying = getVelocityVarying(size);
		setVelocity(velVarying);
	}
}
torch::Tensor FixedBoundary::getVelocity(const bool computational) const {
	TORCH_CHECK(hasSize(), "FixedBoundary has no size set.");
	if(computational && hasTransform()){
		if(m_velocityStatic){
			torch::Tensor velVarying = getVelocityVarying(nullopt);
			return TransformVectors(velVarying, m_transform.value(), true);
		} else {
			return TransformVectors(m_velocity, m_transform.value(), true);
		}
	} else {
		if(m_velocityStatic){
			return getVelocityVarying(nullopt);
		} else {
			return m_velocity;
		}
	}
}

torch::Tensor FixedBoundary::GetFluxes() const {
	TORCH_CHECK(hasSize(), "FixedBoundary has no size set.");
	
	optional<torch::Tensor> fluxes = nullopt;
	if(hasTransform()){
		torch::Tensor vel = getVelocityVarying(nullopt);
		torch::Tensor det = m_transform.value().index({torch::indexing::Ellipsis, -1});
		fluxes = torch::unsqueeze(det, 1) * TransformVectors(vel, m_transform.value(), true);
	} else {
		fluxes = getVelocityVarying(nullopt);
	}
	
	fluxes = fluxes.value().index({torch::indexing::Slice(), torch::indexing::Slice(m_axis,m_axis+1), torch::indexing::Ellipsis});
	
	return fluxes.value().clone();
}

void FixedBoundary::setPressure(torch::Tensor &t){
	index_t spatialDims = getSpatialDims();
	CHECK_INPUT_CUDA(t);
	TORCH_CHECK(t.size(0)==1, "Batches are not yet supported (pressure).");
	TORCH_CHECK(t.size(1)==1, "Pressure channels must be 1.");
	TORCH_CHECK(t.dim()==2 || t.dim()==(2+spatialDims),
		"Pressure spatial dimensions are invalid. Pressure must be " + std::to_string(spatialDims) + "D and either static (shape NC) or varying (shape NCDHW).");
	TORCH_CHECK(t.scalar_type()==getDtype(), "Pressure dtype must match domain.");
	bool pressureStatic = t.dim()==2;
	if(!pressureStatic && hasSize()){
		TORCH_CHECK(checkTensorSpatialSize(t, getSizes(), true), "New pressure spatial dimensions must match existing fields.");
	}
	
	m_pressure = t;
	m_pressureStatic = pressureStatic;
	isTensorChanged=true;
}
void FixedBoundary::setPressureType(const BoundaryConditionType pressureType){
	TORCH_CHECK(pressureType==BoundaryConditionType::NEUMANN, "Invalid pressure boundary type: Currently only Neumann boundaries are supported");
	m_pressureType = pressureType;
}
void FixedBoundary::CreatePressure(const bool createStatic){
	torch::TensorOptions valueOptions = torch::TensorOptions().dtype(getDtype()).layout(torch::kStrided).device(getDevice().type(), getDevice().index());
	torch::Tensor pressure;
	if(createStatic){
		pressure = torch::zeros({1, 1}, valueOptions);
	} else {
		TORCH_CHECK(hasSize(), "FixedBoundary is missing size to create a varying Pressure.");
		const I4 size = getSizes();
		pressure = CreateTensor(1, 1, size, getSpatialDims(), valueOptions);
	}
	setPressure(pressure);
}


void FixedBoundary::setPassiveScalar(torch::Tensor &ps){
	const index_t spatialDims = getSpatialDims();
	CHECK_INPUT_CUDA(ps);
	TORCH_CHECK(ps.size(0)==1, "Batches are not yet supported (passiveScalar).");
	TORCH_CHECK(ps.size(1)==getPassiveScalarChannels(), "Passive scalar channels must match domain.");
	TORCH_CHECK(ps.dim()==2 || ps.dim()==(2+spatialDims), "Passive Scalar spatial dimensions are invalid. Passive Scalar must be 1-3D and either static (shape NC) or varying (shape NCDHW). Varying dimensionality must match velocity.");
	TORCH_CHECK(ps.scalar_type()==getDtype(), "Passive Scalar dtype must match domain.");
	bool passiveScalarStatic = ps.dim()==2;
	
	if(!passiveScalarStatic){
		if(hasSize()){
			TORCH_CHECK(checkTensorSpatialSize(ps, getSizes(), true), "New passive Scalar spatial dimensions must match existing fields.");
		} else {
			setSizeFromTensor(ps, true);
		}
	}
	
	m_passiveScalar = ps;
	m_passiveScalarStatic = passiveScalarStatic;
	
	// keep gradient coherent
	if(m_passiveScalar_grad){
		if(!(m_passiveScalar_grad.value().dim()==ps.dim())){
			// TODO if grad is static: broadcast to varying, else: sum to static
			TORCH_CHECK(false, "New passive scalar does not match existing gradient tensor.");
		}
	}
	
	isTensorChanged=true;
}
void FixedBoundary::CreatePassiveScalar(const bool createStatic){
	const index_t passiveScalarChannels = getPassiveScalarChannels();
	torch::TensorOptions valueOptions = torch::TensorOptions().dtype(getDtype()).layout(torch::kStrided).device(getDevice().type(), getDevice().index());
	torch::Tensor passiveScalar;
	if(createStatic){
		passiveScalar = torch::zeros({1, passiveScalarChannels}, valueOptions);
	} else {
		TORCH_CHECK(hasSize(), "FixedBoundary is missing size to create a varying PassiveScalar.");
		const I4 size = getSizes();
		passiveScalar = CreateTensor(1, passiveScalarChannels, size, getSpatialDims(), valueOptions);
	}
	setPassiveScalar(passiveScalar);
}
void FixedBoundary::setPassiveScalarType(const BoundaryConditionType passiveScalarType) {
	//TORCH_CHECK(passiveScalarType==BoundaryConditionType::DIRICHLET, "Invalid passive scalar boundary type: Currently only Dirichlet boundaries are supported");
	//TORCH_CHECK(passiveScalarType==m_velocityType, "Invalid passive scalar boundary type: Currently must match velocity boundary type.");
	//m_passiveScalarType = passiveScalarType;
	setPassiveScalarType(std::vector<BoundaryConditionType>{passiveScalarType});
}
void FixedBoundary::setPassiveScalarType(const std::vector<BoundaryConditionType> passiveScalarTypes) {
	TORCH_CHECK(hasPassiveScalar(), "FixedBoundary has not passive scalar.");
	TORCH_CHECK(passiveScalarTypes.size()==1 || static_cast<index_t>(passiveScalarTypes.size())==getPassiveScalarChannels(), "Passive scalar boundary conditions must be static or match channels.")

	m_passiveScalarTypes = passiveScalarTypes;

	if(!isPassiveScalarBoundaryTypeStatic()){
		torch::TensorOptions options = torch::TensorOptions().dtype(torch_kBoundaryType).layout(torch::kStrided).device(getDevice().type(), getDevice().index());
		m_passiveScalarTypes_tensor = torch::empty(passiveScalarTypes.size(), options);
		CopyToGPU(m_passiveScalarTypes_tensor.value().data_ptr<BoundaryConditionType_base_type>(), m_passiveScalarTypes.value().data(), passiveScalarTypes.size()*sizeof(BoundaryConditionType));
	}

	isTensorChanged=true;
}
/* void FixedBoundary::clearPassiveScalar() {
	if(hasPassiveScalar()){
		m_passiveScalar = nullopt;
		m_passiveScalarTypes = nullopt;
		m_passiveScalarTypes_tensor=nullopt;
		if(m_passiveScalar_grad){ m_passiveScalar_grad = nullopt; }
		isTensorChanged = true;
	}
} */

#ifdef WITH_GRAD
void FixedBoundary::setPassiveScalarGrad(torch::Tensor &t){
	TORCH_CHECK(m_passiveScalar, "FixedBoundary does not have a passive scalar.");
	CHECK_INPUT_CUDA(t);
	CheckTensor(t, getPassiveScalarChannels(), m_passiveScalar.value(), "FixedBoundary.passiveScalar_grad");
	
	m_passiveScalar_grad = t;
	isTensorChanged=true;
}

void FixedBoundary::CreatePassiveScalarGrad(){
	TORCH_CHECK(m_passiveScalar, "FixedBoundary does not have a passive scalar.");
	torch::Tensor passiveScalar = torch::zeros_like(m_passiveScalar.value());
	setPassiveScalarGrad(passiveScalar);
}

#endif //WITH_GRAD
void FixedBoundary::setTransform(torch::Tensor &t){
	const index_t spatialDims = getSpatialDims();
	CHECK_INPUT_CUDA(t);
	TORCH_CHECK(t.size(0)==1, "Batches are not yet supported (transform).");
	TORCH_CHECK(t.dim()==(2+spatialDims), "Transform spatial dimensions are invalid. Transform must be 1-3D and have shape NDHWT. Dimensionality must match velocity.");
	
	if(hasSize()){
		TORCH_CHECK(checkTensorSpatialSize(t, getSizes(), false), "Transform spatial dimensions must match existing fields.");
	} else {
		setSizeFromTensor(t, false);
	}
	
	TORCH_CHECK(t.size(-1)==TransformNumValues(spatialDims), "Transform channels must match spatial dimensions");
	
	m_transform = t;
	isTensorChanged=true;
}
void FixedBoundary::clearTransform(){
	if(hasTransform()){
		m_transform = nullopt;
		isTensorChanged = true;
	}
}
index_t FixedBoundary::getDim(const dim_t dim) const {
	if(!m_velocityStatic){
		return m_velocity.size(dim);
	} else if(hasPassiveScalar() && !m_passiveScalarStatic){
		return m_passiveScalar.value().size(dim);
	} else if(hasTransform()){
		return m_transform.value().size(dim-1);
	}
	return 1;
}
index_t FixedBoundary::getAxis(const dim_t dim) const {
	TORCH_CHECK(0<=dim && dim<getSpatialDims(), "Axis index must be within spatial dimensions");
	return getDim(getSpatialDims()+1-dim); //velocity.size(velocity.dim()-1 -dim);
}
/* void FixedBoundary::setSpatialDims(const index_t dims) {
	TORCH_CHECK(m_spatialDims==0, "Internal Error: FixedBoundary already has dimensionality set.");
	TORCH_CHECK(0<dims && dims<4, "FixedBoundary must be 1-3D.");
	m_spatialDims = dims;
} */
/*
index_t FixedBoundary::getSpatialDims() const {
	return m_spatialDims;
	// if(m_velocityStatic){
		// return velocity.size(0);
	// }else{
		// return velocity.dim() - 2;
	// }
}*/
bool FixedBoundary::hasSize() const {
	//return !m_velocityStatic || (hasPassiveScalar() && !m_passiveScalarStatic) || hasTransform();
	return m_size.has_value();
}
void FixedBoundary::setSize(I4 size) {
	TORCH_CHECK(!hasSize(), "Internal Error: FixedBoundary already has a size set.");
	size.w = getSpatialDims();
	bool hasSize1 = false; // 1 spatial dimension of a boundary is always 1
	index_t dim = 0; 
	for(;dim<size.w; ++dim){
		if(size.a[dim]==1) {
			TORCH_CHECK(!hasSize1, "Invalid boundary shape: only one dimension must be 1.");
			hasSize1 = true;
			m_axis = dim;
		} else {
			TORCH_CHECK(size.a[dim]>2, "Invalid boundary shape: grid size must be at least 3.");
		}
	}
	for(; dim<3; ++dim){
		size.a[dim]=1;
	}
	TORCH_CHECK(hasSize1, "Invalid boundary shape: one dimension must be 1.");
	m_size = size;
}
void FixedBoundary::setSizeFromTensor(torch::Tensor &t, bool channelsFirst){
	TORCH_CHECK(t.dim()==(getSpatialDims()+2), "Invalid tensor for boundary shape.");
	setSize(getTensorSpatialSize(t, channelsFirst));
}
I4 FixedBoundary::getSizes() const {
	return m_size.value_or(makeI4(1,1,1,getSpatialDims()));
	
	// switch (getSpatialDims())
	// {
	// case 1:
		// return {{.x=getDim(2), .y=1, .z=1, .w=1}};
		// break;
	// case 2:
		// return {{.x=getDim(3), .y=getDim(2), .z=1, .w=2}};
		// break;
	// case 3:
		// return {{.x=getDim(4), .y=getDim(3), .z=getDim(2), .w=3}};
		// break;
	
	// default:
		// return {{.x=1, .y=1, .z=1, .w=1}};
	// }
}
I4 FixedBoundary::getStrides() const {
	const I4 sizes = getSizes();
	return {{.x=1, .y=sizes.x, .z=sizes.x*sizes.y, .w=sizes.x*sizes.y*sizes.z}};
}
/* void FixedBoundary::setDtypeDeviceFromTensor(torch::Tensor &t) {
	TORCH_CHECK(!m_dtype.has_value() && !m_device.has_value(), "FixedBoundary already has device or dtype set.");
	m_device = t.device();
	m_dtype = t.scalar_type();
}
torch::Dtype FixedBoundary::getDtype() const {
	TORCH_CHECK(m_dtype.has_value(), "FixedBoundary does not have a dtype set.");
	return m_dtype.value();
}
torch::Device FixedBoundary::getDevice() const {
	TORCH_CHECK(m_device.has_value(), "FixedBoundary does not have a device set.");
	return m_device.value();
} */
std::string FixedBoundary::ToString() const {
	std::ostringstream repr;
	repr << BoundaryTypeToString(type) << "(";
	repr << getSpatialDims() << "D";
	repr << ", size=" << (hasSize() ? I4toString(getSizes()) : "?");
	repr << ", vStatic=" << (m_velocityStatic ? "true" : "false");
	repr << ", sChannels=" << getPassiveScalarChannels();
	repr << ", sStatic=" << (m_passiveScalarStatic ? "true" : "false");
	repr << ", transform=" << (hasTransform() ? "true" : "false");
	repr << ")";
	return repr.str();
}
//*/

StaticDirichletBoundary::StaticDirichletBoundary(torch::Tensor &slip, torch::Tensor &velocity, torch::Tensor &passiveScalar, const std::shared_ptr<const Domain> p_parentDomain) 
		: Boundary(BoundaryType::DIRICHLET, p_parentDomain), slip(slip), boundaryVelocity(velocity), boundaryScalar(passiveScalar) {
	CHECK_INPUT_HOST(slip);
	TORCH_CHECK(slip.dim()==1, "slip must be 1D.");
	TORCH_CHECK(slip.size(0)==1, "slip must be a scalar.");
	
	CHECK_INPUT_HOST(velocity);
	TORCH_CHECK(velocity.dim()==1, "velocity must be 1D.");
	TORCH_CHECK(0<velocity.size(0) && velocity.size(0)<4, "velocity must have length 1, 2, or 3.");
	
	CHECK_INPUT_HOST(passiveScalar);
	TORCH_CHECK(passiveScalar.dim()==1, "passiveScalar must be 1D.");
	TORCH_CHECK(passiveScalar.size(0)==1, "passiveScalar must have length 1");
	
	TORCH_CHECK(slip.dtype()==velocity.dtype(), "slip and velocity must have same dtype.");
	TORCH_CHECK(passiveScalar.dtype()==velocity.dtype(), "passiveScalar and velocity must have same dtype.");
}
std::shared_ptr<Boundary> StaticDirichletBoundary::Copy() const {
	torch::Tensor s = slip;
	torch::Tensor bv = boundaryVelocity;
	torch::Tensor bs = boundaryScalar;
	return std::make_shared<StaticDirichletBoundary>(s, bv, bs, getParentDomain());
}
std::shared_ptr<Boundary> StaticDirichletBoundary::Clone() const {
	torch::Tensor s = slip.clone();
	torch::Tensor bv = boundaryVelocity.clone();
	torch::Tensor bs = boundaryScalar.clone();
	return std::make_shared<StaticDirichletBoundary>(s, bv, bs, getParentDomain());
}
index_t StaticDirichletBoundary::getSpatialDims() const {
    return boundaryVelocity.size(0);
}

std::string StaticDirichletBoundary::ToString() const {
	std::ostringstream repr;
	repr << BoundaryTypeToString(type) << "(" << getSpatialDims() << "D)";
	return repr.str();
}

VaryingDirichletBoundary::VaryingDirichletBoundary(torch::Tensor &slip, torch::Tensor &velocity, torch::Tensor &passiveScalar, const std::shared_ptr<const Domain> p_parentDomain)
		: Boundary(BoundaryType::DIRICHLET_VARYING, p_parentDomain), slip(slip), boundaryVelocity(velocity), boundaryScalar(passiveScalar) {
	CHECK_INPUT_HOST(slip);
	TORCH_CHECK(slip.dim()==1, "slip must be 1D.");
	TORCH_CHECK(slip.size(0)==1, "slip must be a scalar.");
	
	CHECK_INPUT_CUDA(velocity);
	index_t spatialDims = velocity.dim() - 2;
	TORCH_CHECK(spatialDims>0 && spatialDims<4, "Only 1D, 2D, and 3D is supported. layout should be NC<spatial-dims>, e.g. NCDHW for 3D.");
	TORCH_CHECK(velocity.size(1)==spatialDims, "The velocity channels must match the spatial dimensions.");
	TORCH_CHECK(velocity.size(0)==1, "Batches are not yet supported (velocity).");
	//numDims = spatialDims + 2;
	
	CHECK_INPUT_CUDA(passiveScalar);
	TORCH_CHECK(passiveScalar.dim()==velocity.dim(), "passiveScalar must have same dimensionality as velocity.");
	TORCH_CHECK(passiveScalar.size(1)==1, "The passiveScalar channels must be 1.");
	TORCH_CHECK(passiveScalar.size(0)==1, "Batches are not yet supported (passiveScalar).");
	for(int dim=2;dim<velocity.dim();++dim){
		TORCH_CHECK(velocity.size(dim)==passiveScalar.size(dim), "spatial dimension must match");
	}
	
	TORCH_CHECK(slip.dtype()==velocity.dtype(), "slip and velocity must have same dtype.");
	TORCH_CHECK(passiveScalar.dtype()==velocity.dtype(), "passiveScalar and velocity must have same dtype.");
}
std::shared_ptr<Boundary> VaryingDirichletBoundary::Copy() const {
	torch::Tensor s = slip;
	torch::Tensor bv = boundaryVelocity;
	torch::Tensor bs = boundaryScalar;
	std::shared_ptr<VaryingDirichletBoundary> newBound = std::make_shared<VaryingDirichletBoundary>(s, bv, bs, getParentDomain());
	if(hasTransform){
		torch::Tensor t = transform;
		newBound->setTransform(t);
	}
	return newBound;
}
std::shared_ptr<Boundary> VaryingDirichletBoundary::Clone() const {
	torch::Tensor s = slip.clone();
	torch::Tensor bv = boundaryVelocity.clone();
	torch::Tensor bs = boundaryScalar.clone();
	std::shared_ptr<VaryingDirichletBoundary> newBound = std::make_shared<VaryingDirichletBoundary>(s, bv, bs, getParentDomain());
	if(hasTransform){
		torch::Tensor t = transform.clone();
		newBound->setTransform(t);
	}
	return newBound;
}
void VaryingDirichletBoundary::setTransform(torch::Tensor &newTransform) {
	CHECK_INPUT_CUDA(newTransform);
	TORCH_CHECK(newTransform.dim()==boundaryVelocity.dim(), "new Transform dimensions must match velocity.");
	TORCH_CHECK(newTransform.size(0)==1, "batches are not yet supported.");
	TORCH_CHECK(newTransform.size(-1)==TransformNumValues(getSpatialDims()), "Transform channels must match spatial dimensions");
	for(int dim=2;dim<boundaryVelocity.dim();++dim){
		TORCH_CHECK(boundaryVelocity.size(dim)==newTransform.size(dim-1), "spatial dimension must match");
	}
	TORCH_CHECK(newTransform.dtype()==boundaryVelocity.dtype(), "Transform and velocity must have same dtype.");
	
	transform = newTransform;
	hasTransform = true;
	isTensorChanged=true;
}
void VaryingDirichletBoundary::clearTransform(){
	hasTransform = false;
	transform = torch::empty(0);
	isTensorChanged=true;
}
/*
bool VaryingDirichletBoundary::CheckDataTensor(const torch::Tensor &tensor, const index_t channels, const std::string &name) const {
	return CheckTensor(tensor, channels, boundaryVelocity, name);
}
torch::Tensor VaryingDirichletBoundary::CreateDataTensor(const index_t channels) const {
	return CreateTensor();
}*/
void VaryingDirichletBoundary::setVelocity(torch::Tensor &t){
	CheckTensor(t, getSpatialDims(), boundaryVelocity, "Velocity");
	boundaryVelocity = t;
	isTensorChanged=true;
}
torch::Tensor VaryingDirichletBoundary::getVelocity(const bool computational) const {
	if(computational && hasTransform){
		return TransformVectors(boundaryVelocity, transform, true);
	}
	return boundaryVelocity;
}
torch::Tensor VaryingDirichletBoundary::getBoundaryFlux() const {
	//torch::Tensor boundaryFluxes = hasTransform ? ComputeFluxes(boundaryVelocity, transform) : boundaryVelocity.clone();
	
	return torch::empty(0);
}
void VaryingDirichletBoundary::setPassiveScalar(torch::Tensor &t){
	CheckTensor(t, 1, boundaryScalar, "PassiveScalar");
	boundaryScalar = t;
	isTensorChanged=true;
}

#ifdef WITH_GRAD
void VaryingDirichletBoundary::setVelocityGrad(torch::Tensor &t){
	CheckTensor(t, getSpatialDims(), boundaryVelocity, "VelocityGrad");
	boundaryVelocity_grad = t;
	isTensorChanged=true;
}
void VaryingDirichletBoundary::CreateVelocityGrad(){
	boundaryVelocity_grad = CreateTensorFromRef(1, getSpatialDims(), boundaryVelocity);
	isTensorChanged=true;
}
void VaryingDirichletBoundary::setPassiveScalarGrad(torch::Tensor &t){
	CheckTensor(t, 1, boundaryScalar, "PassiveScalarGrad");
	boundaryScalar_grad = t;
	isTensorChanged=true;
}
void VaryingDirichletBoundary::CreatePassiveScalarGrad(){
	boundaryScalar_grad = CreateTensorFromRef(1, 1, boundaryScalar);
	isTensorChanged=true;
}
#endif

index_t VaryingDirichletBoundary::getSpatialDims() const {
    return boundaryVelocity.dim() - 2;
}

index_t VaryingDirichletBoundary::getDim(const dim_t dim) const {
    return boundaryVelocity.size(dim);
}
index_t VaryingDirichletBoundary::getAxis(const dim_t dim) const {
	TORCH_CHECK(0<=dim && dim<getSpatialDims(), "Axis index must be within spatial dimensions");
    return boundaryVelocity.size(boundaryVelocity.dim()-1 -dim);
}

I4 VaryingDirichletBoundary::getSizes() const {
	switch (getSpatialDims())
	{
	case 1:
		return {{.x=getDim(2), .y=1, .z=1, .w=1}};
		break;
	case 2:
		return {{.x=getDim(3), .y=getDim(2), .z=1, .w=2}};
		break;
	case 3:
		return {{.x=getDim(4), .y=getDim(3), .z=getDim(2), .w=3}};
		break;
	
	default:
		return {{.x=1, .y=1, .z=1, .w=1}};
	}
}

I4 VaryingDirichletBoundary::getStrides() const {
    const I4 sizes = getSizes();
	return {{.x=1, .y=sizes.x, .z=sizes.x*sizes.y, .w=sizes.x*sizes.y*sizes.z}};
}

std::string VaryingDirichletBoundary::ToString() const {
	std::ostringstream repr;
	repr << BoundaryTypeToString(type) << "(" << getSpatialDims() << "D";
	repr << ", size=" << I4toString(getSizes()) << ", stride=" << I4toString(getStrides());
	repr << ")";
	return repr.str();
}

ConnectedBoundary::ConnectedBoundary(std::weak_ptr<Block> wp_connectedBlock, std::vector<dim_t> &axes, const std::shared_ptr<const Domain> p_parentDomain)
		: Boundary(BoundaryType::CONNECTED_GRID, p_parentDomain), axes(axes), wp_connectedBlock(wp_connectedBlock) {
	
	std::shared_ptr<Block> connectedBlock = getConnectedBlock();
	
	TORCH_CHECK(connectedBlock->getSpatialDims()==static_cast<index_t>(axes.size()), "axes must match spatial dimensions of connectedBlock.");
	const index_t numBounds = connectedBlock->getSpatialDims()*2;
	TORCH_CHECK(0<=axes[0] && axes[0]<numBounds, "axes[0] must be in [0,dim*2].");
	
	if(connectedBlock->getSpatialDims()>1){
		TORCH_CHECK(0<=axes[1] && axes[1]<numBounds, "axes[1] must be in [0,dim*2].");
		TORCH_CHECK(BoundaryIndexToDim(axes[0]) != BoundaryIndexToDim(axes[1]), "axes[1] must be a different axis than axes[0].");
		
		if(connectedBlock->getSpatialDims()>2){
			TORCH_CHECK(0<=axes[2] && axes[2]<numBounds, "axes[2] must be in [0,dim*2].");
			TORCH_CHECK(BoundaryIndexToDim(axes[0]) != BoundaryIndexToDim(axes[2]), "axes[2] must be a different axis than axes[0].");
			TORCH_CHECK(BoundaryIndexToDim(axes[1]) != BoundaryIndexToDim(axes[2]), "axes[2] must be a different axis than axes[1].");
		}
	}
}
std::shared_ptr<Block> ConnectedBoundary::getConnectedBlock() const {
	if(std::shared_ptr<Block> connectedBlock = wp_connectedBlock.lock()) {
		return connectedBlock;
	} else {
		TORCH_CHECK(false, "ConnectedBlock is expired.");
	}
}
//torch::Dtype ConnectedBoundary::getDtype() const {return getConnectedBlock()->getDtype();}
//index_t ConnectedBoundary::getSpatialDims() const {return getConnectedBlock()->getSpatialDims();}

dim_t ConnectedBoundary::getConnectionAxis(const dim_t axis) const {
	TORCH_CHECK(0<=axis && axis<getSpatialDims(), "axis out of bounds.")
	return BoundaryIndexToDim(axes[axis]);
}
dim_t ConnectedBoundary::getConnectionAxisDirection(const dim_t axis) const {
	TORCH_CHECK(0<=axis && axis<getSpatialDims(), "axis out of bounds.")
	return axes[axis]&1;
}

std::string ConnectedBoundary::ToString() const {
	
	std::shared_ptr<Block> connectedBlock = getConnectedBlock();
	
	std::ostringstream repr;
	repr << BoundaryTypeToString(type) << "(to=\"" << connectedBlock->name << "\", face=" << BoundaryIndexToString(axes[0]);
	for(index_t dim=1; dim<connectedBlock->getSpatialDims(); ++dim){
		repr << ", axis" << dim << "=" << BoundaryIndexToString(axes[dim]);
	}
	repr << ")";
	return repr.str();
}

/* I4 ConnectedBoundary::getConnectionVector(){
	I4 connections = makeI4();
	return connections;
} */

void ConnectBlocks(std::shared_ptr<Block> block1, const dim_t face1, std::shared_ptr<Block> block2, const dim_t face2, const dim_t connectedAxis1, const dim_t connectedAxis2){
	TORCH_CHECK(block1->getParentDomain()==block2->getParentDomain(), "The blocks must belong to the same domain.");
	const dim_t spatialDims = block1->getSpatialDims();
	TORCH_CHECK(spatialDims==block2->getSpatialDims(), "The spatial dimensions of the blocks must match.");
	
	std::vector<dim_t> axes1 = {face2};
	std::vector<dim_t> axes2 = {face1};
	if(spatialDims>1){
		axes1.push_back(connectedAxis1);
		const dim_t face1Dim = BoundaryIndexToDim(face1);
		const dim_t face2Dim = BoundaryIndexToDim(face2);
		bool axesSwapped = false;
		if(spatialDims==2 || (BoundaryIndexToDim(connectedAxis1) == (face2Dim + 1)%spatialDims)) {
			axes2.push_back((((face1Dim + 1)%spatialDims)<<1) | (connectedAxis1&1));
			axesSwapped = false;
		} else {
			//spatialDims==3 here
			TORCH_CHECK((connectedAxis2>>1) == (face2Dim+ 1)%spatialDims, "Invalid connection.")
			axes2.push_back((((face1Dim + 2)%spatialDims)<<1) | (connectedAxis2&1));
			axesSwapped = true;
		}
		if(spatialDims>2){
			axes1.push_back(connectedAxis2);
			if(!axesSwapped){
				axes2.push_back((((face1Dim + 2)%spatialDims)<<1) | (connectedAxis2&1));
			} else {
				axes2.push_back((((face1Dim + 1)%spatialDims)<<1) | (connectedAxis1&1));
			}
		}
	}
	std::shared_ptr<ConnectedBoundary> bound1 = std::make_shared<ConnectedBoundary>(block2, axes1, block1->getParentDomain());
	block1->setBoundary(face1, bound1);
	std::shared_ptr<ConnectedBoundary> bound2 = std::make_shared<ConnectedBoundary>(block1, axes2, block2->getParentDomain());
	block2->setBoundary(face2, bound2);
}
void ConnectBlocks(std::shared_ptr<Block> block1, const std::string &face1, std::shared_ptr<Block> block2, const std::string &face2, const std::string &connectedAxis1, const std::string &connectedAxis2){
	ConnectBlocks(block1, BoundarySideToIndex(face1), block2, BoundarySideToIndex(face2), BoundarySideToIndex(connectedAxis1), BoundarySideToIndex(connectedAxis2));
}

Block::Block(optional<torch::Tensor> velocity, optional<torch::Tensor> pressure, optional<torch::Tensor> passiveScalar,
		optional<torch::Tensor> vertexCoordinates, const std::string &name, const std::shared_ptr<const Domain> p_parentDomain) : name(name), wp_parentDomain(p_parentDomain) {
	
	TORCH_CHECK(velocity || pressure || passiveScalar || vertexCoordinates, "At least one field (velocity, pressure, passiveScalar, vertexCoordinates) or the size must be given to create a block.")
	
	const index_t spatialDims = p_parentDomain->getSpatialDims();
	//const index_t passiveScalarChannels = p_parentDomain->getPassiveScalarChannels();
	const torch::Dtype dtype = p_parentDomain->getDtype();
	const torch::Device device = p_parentDomain->getDevice();
	
	//index_t spatialDims = 0;
	if(velocity){
		torch::Tensor velocityTensor = velocity.value();
		CHECK_INPUT_CUDA(velocityTensor);
		//spatialDims = velocityTensor.dim() - 2;
		TORCH_CHECK(velocityTensor.dim()==spatialDims+2, "Velocity dimensionality must match domain. Layout should be NC<spatial-dims>, e.g. NCDHW for 3D.");
		TORCH_CHECK(velocityTensor.size(1)==spatialDims, "The velocity channels must match the spatial dimensions.");
		TORCH_CHECK(velocityTensor.size(0)==1, "Batches are not yet supported.");
		for(int dim=2;dim<velocityTensor.dim();++dim){
			TORCH_CHECK(velocityTensor.size(dim)>2, "all spatial dimensions must be at least 3.");
		}
		//numDims = spatialDims + 2;
		
		this->velocity = velocityTensor;
	} else {
		TensorInfo fieldInfo;
		if(pressure){
			fieldInfo = getFieldInfo(pressure.value(), false);
		} else if(passiveScalar){
			fieldInfo = getFieldInfo(passiveScalar.value(), false);
		} else {
			fieldInfo = getFieldInfo(vertexCoordinates.value(), true);
		}
		
		TORCH_CHECK(fieldInfo.spatialDims==spatialDims, "Field dimensionality must match domain. Layout should be NC<spatial-dims>, e.g. NCDHW for 3D.");
		//spatialDims = fieldInfo.spatialDims;
		//numDims = fieldInfo.dims;
		
		this->velocity = CreateTensor(1, spatialDims, fieldInfo.size, spatialDims, 
			torch::TensorOptions().dtype(fieldInfo.dtype).layout(torch::kStrided).device(fieldInfo.device.value().type(), fieldInfo.device.value().index()));
	}
	
	// set simple default boundaries before anything might access them.
	boundaries.reserve(spatialDims*2);
	for(index_t bound=0; bound<spatialDims*2 ;++bound){
		boundaries.push_back(std::make_shared<PeriodicBoundary>(p_parentDomain));
	}
	
	if(pressure){
		setPressure(pressure.value());
	}else{
		CreatePressure();
	}
	
	if(passiveScalar){
		setPassiveScalar(passiveScalar.value());
	} else if(p_parentDomain->hasPassiveScalar()){
		CreatePassiveScalar();
	}
	
	if(vertexCoordinates){
		setVertexCoordinates(vertexCoordinates.value());
	}
};

//Block::Block(const I4 size, const index_t passiveScalarChannels, const std::string &name, const torch::Dtype dtype, const torch::Device device) : name(name){
Block::Block(const I4 size, const std::string &name, const std::shared_ptr<const Domain> p_parentDomain) : name(name), wp_parentDomain(p_parentDomain) {
	const index_t spatialDims = p_parentDomain->getSpatialDims(); //size.w;
	//const index_t passiveScalarChannels = p_parentDomain->getPassiveScalarChannels();
	const torch::Dtype dtype = p_parentDomain->getDtype();
	const torch::Device device = p_parentDomain->getDevice();
	
	//numDims = spatialDims + 2;
	TORCH_CHECK(0<spatialDims && spatialDims<4, "Only 1D, 2D, and 3D is supported.");
	for(index_t dim=0; dim<spatialDims; ++dim){
		TORCH_CHECK(size.a[dim]>2, "all spatial dimensions must be at least 3.");
	}
	//TORCH_CHECK(device.is_cuda(), "Device must be a CUDA.")
	
	velocity = CreateTensor(1, spatialDims, size, spatialDims, torch::TensorOptions().dtype(dtype).layout(torch::kStrided).device(device.type(), device.index()));
	
	CreatePressure();
	
	if(p_parentDomain->hasPassiveScalar()){
		CreatePassiveScalar();
	}
	
	boundaries.reserve(spatialDims*2);
	for(index_t bound=0; bound<spatialDims*2 ;++bound){
		boundaries.push_back(std::make_shared<PeriodicBoundary>(p_parentDomain));
	}
}

#ifdef DTOR_MSG
Block::~Block(){
	py::print("Block dtor");
	//DetachFwd();
}
#endif

std::shared_ptr<Block> Block::Copy() const {
	torch::Tensor v = velocity;
	torch::Tensor p = pressure;
	optional<torch::Tensor> s = passiveScalar;
	optional<torch::Tensor> c = m_vertexCoordinates;
	const bool hasOnlyTransform = !m_vertexCoordinates && m_transform;
	std::string n = name+"_copy";
	std::shared_ptr<Block> cBlock = std::make_shared<Block>(v, p, s, c, n, getParentDomain());
	if(hasOnlyTransform){
		// cBlock boundaries are default periodic here, so no boundary transform update happens.
		torch::Tensor t = m_transform.value();
		cBlock->setTransform(t, m_faceTransform);
	}
	if(hasVelocitySource()){
		torch::Tensor vs = velocitySource.value();
		cBlock->setVelocitySource(vs);
	}
	if(hasViscosity()){
		torch::Tensor visc = m_viscosity.value();
		cBlock->setViscosity(visc);
	}
	// copy prescibed boudaries. connected is handled on domain level, periodic is default.
	for(index_t i=0;i<static_cast<index_t>(boundaries.size());++i){
		BoundaryType bt = boundaries[i]->type;
		if(bt==BoundaryType::DIRICHLET || bt==BoundaryType::DIRICHLET_VARYING || bt==BoundaryType::FIXED){
			cBlock->setBoundary(i, boundaries[i]->Copy());
		}
	}
	return cBlock;
}
std::shared_ptr<Block> Block::Clone() const {
	torch::Tensor v = velocity.clone();
	torch::Tensor p = pressure.clone();
	optional<torch::Tensor> s = cloneOptionalTensor(passiveScalar);
	optional<torch::Tensor> c = cloneOptionalTensor(m_vertexCoordinates);
	const bool hasOnlyTransform = !m_vertexCoordinates && m_transform;
	std::string n = name+"_clone";
	std::shared_ptr<Block> cBlock = std::make_shared<Block>(v, p, s, c, n, getParentDomain());
	if(hasOnlyTransform){
		// cBlock boundaries are default periodic here, so no boundary transform update happens.
		torch::Tensor t = m_transform.value().clone();
		cBlock->setTransform(t, cloneOptionalTensor(m_faceTransform));
	}
	if(hasVelocitySource()){
		torch::Tensor vs = velocitySource.value().clone();
		cBlock->setVelocitySource(vs);
	}
	if(hasViscosity()){
		torch::Tensor visc = m_viscosity.value().clone();
		cBlock->setViscosity(visc);
	}
	// copy prescibed boudaries. connected is handled on domain level, periodic is default.
	for(index_t i=0;i<static_cast<index_t>(boundaries.size());++i){
		BoundaryType bt = boundaries[i]->type;
		if(bt==BoundaryType::DIRICHLET || bt==BoundaryType::DIRICHLET_VARYING || bt==BoundaryType::FIXED){
			cBlock->setBoundary(i, boundaries[i]->Clone());
		}
	}
	return cBlock;
}


std::shared_ptr<const Domain> Block::getParentDomain() const {
	if(std::shared_ptr<const Domain> parentDomain = wp_parentDomain.lock()) {
		return parentDomain;
	} else {
		TORCH_CHECK(false, "Parent Domain is expired.");
	}
}

void Block::DetachFwd() {
	velocity = velocity.detach();
	if(velocitySource){ velocitySource = velocitySource.value().detach(); }
	pressure = pressure.detach();
	if(hasPassiveScalar()) { passiveScalar = passiveScalar.value().detach(); }
	if(m_vertexCoordinates){ m_vertexCoordinates = m_vertexCoordinates.value().detach(); }
	if(m_transform){ m_transform = m_transform.value().detach(); }
	if(m_faceTransform){ m_faceTransform = m_faceTransform.value().detach(); }
	
	for(const auto &bound : getFixedBoundaries()){
		bound.second->DetachFwd();
	}
}
void Block::DetachGrad() {
	velocity_grad = velocity_grad.detach();
	if(velocitySource_grad){ velocitySource_grad = velocitySource_grad.value().detach(); }
	if(!IsTensorEmpty(pressure_grad)){ pressure_grad = pressure_grad.detach(); }
	if(hasPassiveScalar()) { passiveScalar_grad = passiveScalar_grad.detach(); }
	
	for(const auto &bound : getFixedBoundaries()){
		bound.second->DetachGrad();
	}
}
void Block::Detach() {
	DetachFwd();
	DetachGrad();
}


torch::Dtype Block::getDtype() const{
	return getParentDomain()->getDtype();
}
torch::Device Block::getDevice() const{
	return getParentDomain()->getDevice();
}

torch::TensorOptions Block::getValueOptions() const {
	//return torch::TensorOptions().dtype(getDtype()).layout(torch::kStrided).device(getDevice().type(), getDevice().index());
	return getParentDomain()->getValueOptions();
}

bool Block::CheckDataTensor(const torch::Tensor &tensor, const index_t channels, const bool allowStatic, const std::string &name) const {
	CHECK_INPUT_CUDA(tensor);
	const index_t numDims = getSpatialDims() + 2;
	TORCH_CHECK(tensor.dim()==numDims || (allowStatic && tensor.dim()==2), "Dimensions of " + name + " must be " + std::to_string(numDims) + (allowStatic ? " or 2." : "."));
	TORCH_CHECK(tensor.size(0)==1, "Batches (dim 0) are not yet supported (batch dimension must be 1).");
	if(channels<1){
		// free number of channels
	}else{
		TORCH_CHECK(tensor.size(1)==channels, "Channels (dim 1) of " + name + " must be " + std::to_string(channels) + ".");
	}
	if(tensor.dim()!=2){
		for(int dim=2;dim<numDims;++dim){
			TORCH_CHECK(velocity.size(dim)==tensor.size(dim), "Spatial dimension " + std::to_string(dim) + " of " + name +
				" must match (" + std::to_string(velocity.size(dim)) + "), but is (" + std::to_string(tensor.size(dim)) + ").");
		}
	}
	TORCH_CHECK(tensor.dtype()==velocity.dtype(), name + "has wrong dtype.");
	return true;
}

torch::Tensor Block::CreateDataTensor(const index_t channels) const {
	
	return CreateTensor(1, channels, getSizes(), getSpatialDims(), getValueOptions());
}

void Block::setVelocity(torch::Tensor &v){
	CheckDataTensor(v, getSpatialDims(), false, "Velocity");
	velocity = v;
	isTensorChanged=true;
}
torch::Tensor Block::getVelocity(const bool computational) const {
	if(computational){
		TORCH_CHECK(hasTransform(), "Coordinates or Transformations are required to compute computational velocities.");
		return TransformVectors(velocity, m_transform.value(), true);
	}
	return velocity;
}

void Block::setVelocitySource(torch::Tensor &t){
	/* index_t spatialDims = getSpatialDims();
	CHECK_INPUT_CUDA(t);
	TORCH_CHECK(t.size(0)==1, "Batches are not yet supported (velocity source).");
	TORCH_CHECK(t.size(1)==spatialDims, "Velocity source channels are invalid. Velocity source must be " + std::to_string(spatialDims) + "D and either static (shape NC) or varying (shape NCDHW).");
	TORCH_CHECK(t.dim()==2 || t.dim()==(2+spatialDims),
		"Velocity source spatial dimensions are invalid. Velocity source must be " + std::to_string(spatialDims) + "D and either static (shape NC) or varying (shape NCDHW).");
	bool velocityStatic = t.dim()==2;
	if(!velocityStatic){
		TORCH_CHECK(checkTensorSpatialSize(t, getSizes(), true), "New velocity spatial dimensions must match existing fields.");
	} */
	
	CheckDataTensor(t, getSpatialDims(), true, "VelocitySource");
	
	velocitySource = t;
	velocitySourceStatic = t.dim()==2; //velocityStatic;
	
	// keep gradient coherent
	if(velocitySource_grad){
		if(!(velocitySource_grad.value().dim()==t.dim())){
			// TODO if grad is static: broadcast to varying, else: sum to static
			TORCH_CHECK(false, "New velocity source does not match existing gradient tensor.");
		}
	}

	isTensorChanged = true;
}
void Block::CreateVelocitySource(const bool createStatic){
	torch::TensorOptions valueOptions = getValueOptions();
	torch::Tensor velocity;
	if(createStatic){
		velocity = torch::zeros({1, getSpatialDims()}, valueOptions);
	} else {
		const I4 size = getSizes();
		velocity = CreateTensor(1, getSpatialDims(), size, getSpatialDims(), valueOptions);
	}
	setVelocitySource(velocity);
}
void Block::clearVelocitySource(){
	if(hasVelocitySource()){
		velocitySource = nullopt;
		isTensorChanged = true;
	}
}


void Block::setViscosity(const torch::Tensor &v){
	CheckDataTensor(v, 1, true, "BlockViscosity");
	
	if(m_viscosity_grad){
		if(!(m_viscosity_grad.value().dim()==v.dim())){
			// TODO if grad is static: broadcast to varying, else: sum to static
			TORCH_CHECK(false, "New block viscosity does not match existing gradient tensor.");
		}
	}
	
	m_viscosity = v;
	isTensorChanged = true;
}
void Block::clearViscosity(){
	if(hasViscosity()){
		m_viscosity = nullopt;
		isTensorChanged = true;
	}
	
}

void Block::setPressure(torch::Tensor &p){
	CheckDataTensor(p, 1, false, "Pressure");
	pressure = p;
	isTensorChanged=true;
}
void Block::setPassiveScalar(torch::Tensor &s){
	const index_t channels = getPassiveScalarChannels();
	TORCH_CHECK(channels>0, "Passive scalars are not active.");
	CheckDataTensor(s, channels, false, "Passive Scalar");
	passiveScalar = s;
	isTensorChanged=true;
}
void Block::CreatePassiveScalar(){
	const index_t channels = getPassiveScalarChannels();
	TORCH_CHECK(channels>0, "Passive scalars are not active.");
	torch::Tensor sg = CreateDataTensor(channels);
	setPassiveScalar(sg);
}
/* void Block::clearPassiveScalar(){
	if(passiveScalar){
		passiveScalar = nullopt;
		isTensorChanged=true;
	}
} */
/* template <typename scalar_t>
scalar_t* Block::getPassiveScalarDataPtr() const {
	return getOptionalTensorDataPtr(passiveScalar);
} */

bool Block::hasPassiveScalar() const {
	return getParentDomain()->hasPassiveScalar();
}
index_t Block::getPassiveScalarChannels() const {
	//return hasPassiveScalar() ? passiveScalar.value().size(1) : 0;
	return getParentDomain()->getPassiveScalarChannels();
}

void Block::CreatePressure(){
	torch::Tensor p = CreateDataTensor(1);
	setPressure(p);
}
void Block::CreateVelocity(){
	torch::Tensor v = CreateDataTensor(getSpatialDims());
	setVelocity(v);
}

#ifdef WITH_GRAD
void Block::setVelocityGrad(torch::Tensor &vg){
	CheckDataTensor(vg, getSpatialDims(), false, "VelocityGrad");
	velocity_grad = vg;
	isTensorChanged=true;
}
void Block::CreateVelocityGrad(){
	torch::Tensor vg = CreateDataTensor(getSpatialDims());
	setVelocityGrad(vg);
}

void Block::setVelocitySourceGrad(torch::Tensor &t){
	CHECK_INPUT_CUDA(t);
	TORCH_CHECK(velocitySource.has_value(), "Block does not have a velocity source tensor");
	index_t spatialDims = getSpatialDims();
	CheckTensor(t, spatialDims, velocitySource.value(), "Block.velocitySource_grad");
	
	velocitySource_grad = t;
	isTensorChanged=true;

}
void Block::CreateVelocitySourceGrad(){
	if(velocitySource){
		torch::Tensor velocity_grad = torch::zeros_like(velocitySource.value());
		setVelocitySourceGrad(velocity_grad);
	} else {
		clearVelocitySourceGrad();
	}
}
void Block::clearVelocitySourceGrad(){
	if(hasVelocitySourceGrad()){
		velocitySource_grad = nullopt;
		isTensorChanged = true;
	}
}

void Block::setViscosityGrad(const torch::Tensor &t){
	CHECK_INPUT_CUDA(t);
	TORCH_CHECK(m_viscosity.has_value(), "Block does not have a viscosity tensor");
	index_t spatialDims = getSpatialDims();
	CheckTensor(t, 1, m_viscosity.value(), "Block.viscosity_grad");
	
	m_viscosity_grad = t;
	isTensorChanged=true;
}
void Block::CreateViscosityGrad(){
	if(m_viscosity){
		torch::Tensor viscosity_grad = torch::zeros_like(m_viscosity.value());
		setViscosityGrad(viscosity_grad);
	} else {
		clearViscosityGrad();
	}
}
void Block::clearViscosityGrad(){
	if(hasViscosityGrad()){
		m_viscosity_grad = nullopt;
		isTensorChanged = true;
	}
}

void Block::setPressureGrad(torch::Tensor &pg){
	CheckDataTensor(pg, 1, false, "PressureGrad");
	pressure_grad= pg;
	isTensorChanged=true;
}
void Block::CreatePressureGrad(){
	torch::Tensor pg = CreateDataTensor(1);
	setPressureGrad(pg);
}
void Block::setPassiveScalarGrad(torch::Tensor &sg){
	CheckDataTensor(sg, getPassiveScalarChannels(), false, "PassiveScalarGrad");
	passiveScalar_grad = sg;
	isTensorChanged=true;
}
void Block::CreatePassiveScalarGrad(){
	torch::Tensor sg = CreateDataTensor(getPassiveScalarChannels());
	setPassiveScalarGrad(sg);
}


void Block::CreatePassiveScalarGradOnBoundaries(){
	for(auto boundary : boundaries){
		if(boundary->type==BoundaryType::FIXED){
			std::shared_ptr<FixedBoundary> bound = std::static_pointer_cast<FixedBoundary> (boundary);
			bound->CreatePassiveScalarGrad();
		}
	}
}
void Block::CreateVelocityGradOnBoundaries(){
	for(auto boundary : boundaries){
		if(boundary->type==BoundaryType::FIXED){
			std::shared_ptr<FixedBoundary> bound = std::static_pointer_cast<FixedBoundary> (boundary);
			bound->CreateVelocityGrad();
		}
	}
}

#endif //WITH_GRAD

torch::Tensor Block::getMaxVelocity(const bool withBounds, const bool computational) const{
	torch::Tensor maxVel = torch::max(torch::abs(getVelocity(computational)));
	
	if(withBounds){
		for(auto boundary : boundaries){
			switch(boundary->type){
				case BoundaryType::DIRICHLET:
				{
					std::shared_ptr<StaticDirichletBoundary> bound = std::static_pointer_cast<StaticDirichletBoundary> (boundary);
					maxVel = torch::maximum(maxVel, torch::max(torch::abs(bound->boundaryVelocity)));
					break;
				}
				case BoundaryType::DIRICHLET_VARYING:
				{
					std::shared_ptr<VaryingDirichletBoundary> bound = std::static_pointer_cast<VaryingDirichletBoundary> (boundary);
					maxVel = torch::maximum(maxVel, torch::max(torch::abs(bound->getVelocity(computational))));
					break;
				}
				case BoundaryType::FIXED:
				{
					std::shared_ptr<FixedBoundary> bound = std::static_pointer_cast<FixedBoundary> (boundary);
					maxVel = torch::maximum(maxVel, torch::max(torch::abs(bound->getVelocity(computational))));
					break;
				}
				default:
					break;
			}
		}
	}
	
	return maxVel;
}

torch::Tensor getVelocityMagnitude(const torch::Tensor &velocity){ //, const index_t spatialDims, const index_t totalSize){
	//return torch::linalg::vector_norm(velocity, 2, 1, true, c10::nullopt);
	return at::linalg_vector_norm(velocity, 2, 1, true);
}

torch::Tensor Block::getMaxVelocityMagnitude(const bool withBounds, const bool computational) const{
	torch::Tensor maxMag = torch::max(getVelocityMagnitude(getVelocity(computational)));
	
	if(withBounds){
		for(auto boundary : boundaries){
			switch(boundary->type){
				case BoundaryType::DIRICHLET:
				{
					std::shared_ptr<StaticDirichletBoundary> bound = std::static_pointer_cast<StaticDirichletBoundary> (boundary);
					maxMag = torch::maximum(maxMag, torch::max(getVelocityMagnitude(bound->boundaryVelocity)));
					break;
				}
				case BoundaryType::DIRICHLET_VARYING:
				{
					std::shared_ptr<VaryingDirichletBoundary> bound = std::static_pointer_cast<VaryingDirichletBoundary> (boundary);
					maxMag = torch::maximum(maxMag, torch::max(getVelocityMagnitude(bound->getVelocity(computational))));
					break;
				}
				case BoundaryType::FIXED:
				{
					std::shared_ptr<FixedBoundary> bound = std::static_pointer_cast<FixedBoundary> (boundary);
					maxMag = torch::maximum(maxMag, torch::max(getVelocityMagnitude(bound->getVelocity(computational))));
					break;
				}
				default:
					break;
			}
		}
	}
	
	return maxMag;
}


bool Block::IsValidFaceIndex(const index_t face) const {
	return 0 <= face && face < (getSpatialDims()*2);
}
bool Block::IsValidAxisIndex(const index_t axis) const {
	return 0 <= axis && axis < getSpatialDims();
}
void Block::CheckFaceIndex(const index_t face) const {
	TORCH_CHECK(IsValidFaceIndex(face), "Face index must be in [0, dims*2) or one of {-x, +x, -y, +y, -z, +z} for exising spatial dimensions.");
}
void Block::CheckAxisIndex(const index_t axis) const {
	TORCH_CHECK(IsValidAxisIndex(axis), "Axis index must be in [0, dims) or one of {x, y, z} for exising spatial dimensions.");
}

void Block::setBoundary(const std::string &side, std::shared_ptr<Boundary> boundary){
    setBoundary(BoundarySideToIndex(side), boundary);
};

void Block::setBoundary(const index_t index, std::shared_ptr<Boundary> boundary){
	CheckFaceIndex(index);
	switch(boundary->type){
	case BoundaryType::FIXED:
		{
			std::shared_ptr<FixedBoundary> bound = std::static_pointer_cast<FixedBoundary> (boundary);
			const index_t spatialDims = getSpatialDims();
			TORCH_CHECK(spatialDims==bound->getSpatialDims(), "Spatial dimensions of block and boundary must match.");
			TORCH_CHECK(getPassiveScalarChannels()==bound->getPassiveScalarChannels(), "Passive scalar channels of block and boundary must match.");
			TORCH_CHECK(getDtype()==bound->getDtype(), "Dtype of block and boundary must match.");
			TORCH_CHECK(getDevice()==bound->getDevice(), "Device of block and boundary must match.");
			if(bound->hasSize()){
				index_t axis = (index>>1);
				TORCH_CHECK(bound->getAxis(axis)==1, "Boundary axis size must be 1.");
				if(spatialDims>1){
					axis = ((index>>1) + 1)%spatialDims;
					TORCH_CHECK(bound->getAxis(axis)==getAxis(axis), "Boundary size must match block.");
					if(spatialDims>2){
						axis = ((index>>1) + 2)%spatialDims;
						TORCH_CHECK(bound->getAxis(axis)==getAxis(axis), "Boundary size must match block.");
					}
				}
			} else {
				I4 boundSize = getSizes();
				boundSize.a[index>>1] = 1;
				bound->setSize(boundSize);
			}
			if(hasTransform()!=bound->hasTransform()){
				py::print("Warning: Only one of Block and FixedBoundary has a transformation set.");
			}
		}
		break;
	case BoundaryType::DIRICHLET:
		{
			std::shared_ptr<StaticDirichletBoundary> bound = std::static_pointer_cast<StaticDirichletBoundary> (boundary);
			TORCH_CHECK(getSpatialDims()==bound->getSpatialDims(), "Spatial dimensions of block and boundary must match.");
			TORCH_CHECK(getDtype()==bound->getDtype(), "Dtype of block and boundary must match.");
		}
		break;
	case BoundaryType::DIRICHLET_VARYING:
		{
			std::shared_ptr<VaryingDirichletBoundary> bound = std::static_pointer_cast<VaryingDirichletBoundary> (boundary);
			const index_t spatialDims = getSpatialDims();
			TORCH_CHECK(spatialDims==bound->getSpatialDims(), "Spatial dimensions of block and boundary must match.");
			TORCH_CHECK(getDtype()==bound->getDtype(), "Dtype of block and boundary must match.");
			//check each dimension/axis
			index_t axis = (index>>1);
			TORCH_CHECK(bound->getAxis(axis)==1, "");
			if(spatialDims>1){
				axis = ((index>>1) + 1)%spatialDims;
				TORCH_CHECK(bound->getAxis(axis)==getAxis(axis), "");
				if(spatialDims>2){
					axis = ((index>>1) + 2)%spatialDims;
					TORCH_CHECK(bound->getAxis(axis)==getAxis(axis), "");
				}
			}
			if(hasTransform()!=bound->hasTransform){
				py::print("Warning: Only one of Block and VaryingDirichletBoundary has a transformation set.");
			}
		}
		break;
	case BoundaryType::CONNECTED_GRID:
		{
			std::shared_ptr<ConnectedBoundary> bound = std::static_pointer_cast<ConnectedBoundary> (boundary);
			std::shared_ptr<Block> otherBlock = bound->getConnectedBlock();
			const index_t spatialDims = getSpatialDims();
			TORCH_CHECK(spatialDims==otherBlock->getSpatialDims(), "Spatial dimensions of blocks must match.");
			TORCH_CHECK(getDtype()==otherBlock->getDtype(), "Dtype of blocks must match.");
			
			//axis index: 0,1,2 -> x,y,z
			// getDim: 3D [0,4] -> NCzyx
			//const index_t maxDim = spatialDims + 1;
			if(spatialDims>1){
				dim_t axisIndex = ((index>>1)+1)%spatialDims;
				dim_t otherAxisIndex = bound->getConnectionAxis(1);
				TORCH_CHECK(getAxis(axisIndex)==otherBlock->getAxis(otherAxisIndex), "First connection axis size does not match.")
				if(spatialDims>2){
					axisIndex = ((index>>1)+2)%spatialDims;
					otherAxisIndex = bound->getConnectionAxis(2);
					TORCH_CHECK(getAxis(axisIndex)==otherBlock->getAxis(otherAxisIndex), "Second connection axis size does not match.")
				}
			}
		}
		break;
	default:
		break;
	}
	boundaries[index] = boundary;
}

std::shared_ptr<Boundary> Block::getBoundary(const std::string &side) const {
    return getBoundary(BoundarySideToIndex(side));
}

std::shared_ptr<Boundary> Block::getBoundary(const index_t index) const {
	TORCH_CHECK(index>=0 && index<getSpatialDims()*2, "Invalid boundary location specified.");
    std::shared_ptr<Boundary> bound = boundaries[index];
	return bound;
}

std::vector<std::pair<index_t, std::shared_ptr<FixedBoundary>>> Block::getFixedBoundaries() const {
	std::vector<std::pair<index_t, std::shared_ptr<FixedBoundary>>> bounds;
	for(index_t boundIdx=0; boundIdx<(getSpatialDims()*2); ++boundIdx){
		if(boundaries[boundIdx]->type==BoundaryType::FIXED){
			std::pair<index_t, std::shared_ptr<FixedBoundary>> bound = std::make_pair(boundIdx, std::static_pointer_cast<FixedBoundary>(boundaries[boundIdx]));
			bounds.push_back(bound);
		}
	}
	return bounds;
}

bool Block::isAllFixedBoundariesPassiveScalarTypeStatic() const {
	for(const auto &bound : getFixedBoundaries()){
		if(!bound.second->isPassiveScalarBoundaryTypeStatic()){
			return false;
		}
	}
	return true;
}

void Block::CloseConnectedBoudary(const index_t face, const bool useFixed){
	CheckFaceIndex(face);
	std::shared_ptr<Boundary> boundary = getBoundary(face);
	switch(boundary->type){
		case BoundaryType::CONNECTED_GRID:
		{
			std::shared_ptr<ConnectedBoundary> bound = std::static_pointer_cast<ConnectedBoundary> (boundary);
			std::shared_ptr<Block> otherBlock = bound->getConnectedBlock();
			const index_t otherBoundIndex = bound->axes[0];
			// std::shared_ptr<Boundary> otherboundary = otherBlock->getBoundary(otherBoundIndex);
			// if(otherboundary->type == BoundaryType::CONNECTED_GRID){
				// std::shared_ptr<ConnectedBoundary> otherBound = std::static_pointer_cast<ConnectedBoundary> (boundary);
				// if(otherBound->getConnectedBlock() == shared_from_this() && otherBound->axes[0] == static_cast<dim_t>(face)){
					if(useFixed){
						otherBlock->MakeFixedBoundary(otherBoundIndex, nullopt, BoundaryConditionType::DIRICHLET, nullopt, nullopt);
					} else {
						otherBlock->MakeClosedBoundary(otherBoundIndex);
					}
				// }
			// }
			break;
		}
		case BoundaryType::PERIODIC:
		{
			if(useFixed){
				MakeFixedBoundary(face ^ 1, nullopt, BoundaryConditionType::DIRICHLET, nullopt, nullopt);
			}else {
				MakeClosedBoundary(face ^ 1);
			}
			break;
		}
		default:
			break;
	}
}

torch::Tensor Block::GetFaceTransformBoundarySlice(const index_t face) const {
	CheckFaceIndex(face);
	TORCH_CHECK(hasFaceTransform(), "faceTransform missing to slice for boundary.");
	const index_t axis = face >> 1;
	const bool isUpper = face & 1;
	
	std::vector<torch::indexing::TensorIndex> slicing;
	slicing.push_back(torch::indexing::Slice()); // batch, no change
	//slicing.push_back(torch::indexing::Slice(axis, axis+1)); // channel, get for axis, but keep dimension
	slicing.push_back(axis); // channel, get for axis
	for(index_t dim=getSpatialDims()-1; dim>=0; --dim){
		if(dim==axis){
			if(isUpper){
				slicing.push_back(torch::indexing::Slice(-1,torch::indexing::None));
			} else {
				slicing.push_back(torch::indexing::Slice(0,1));
			}
		} else {
			slicing.push_back(torch::indexing::Slice(0,-1)); // remove excess top from staggeted grid
		}
	}
	slicing.push_back(torch::indexing::Slice()); // transform struct size, no change
	
	torch::Tensor boundaryTransform = m_faceTransform.value().index(slicing).clone().contiguous();
	
	return boundaryTransform;
}

void Block::UpdateBoundaryTransforms(){
	if(!hasFaceTransform()) { return; }
	for(index_t bound=0; bound<(getSpatialDims()*2); ++bound){
		std::shared_ptr<Boundary> boundary = getBoundary(bound);
		switch(boundary->type){
			case BoundaryType::DIRICHLET:
			{
				TORCH_CHECK(false, "TODO: Dirichlet->DirichletVarying on transform update?");
				break;
			}
			case BoundaryType::DIRICHLET_VARYING:
			{
				std::shared_ptr<VaryingDirichletBoundary> vdb = std::static_pointer_cast<VaryingDirichletBoundary> (boundary);
				torch::Tensor boundaryTransform = GetFaceTransformBoundarySlice(bound);
				vdb->setTransform(boundaryTransform);
				break;
			}
			case BoundaryType::FIXED:
			{
				std::shared_ptr<FixedBoundary> fb = std::static_pointer_cast<FixedBoundary> (boundary);
				torch::Tensor boundaryTransform = GetFaceTransformBoundarySlice(bound);
				fb->setTransform(boundaryTransform);
				break;
			}
			default:
				break;
		}
	}
}

void Block::MakeClosedBoundary(const index_t bound){
	TORCH_CHECK(false, "Old boundary formats are no longer supported.");
	CheckFaceIndex(bound);
	torch::TensorOptions CPUValueOptions = torch::TensorOptions().dtype(getDtype()).layout(torch::kStrided);
	torch::Tensor boundarySlip = torch::zeros({1}, CPUValueOptions);
	if(hasFaceTransform()){
		torch::TensorOptions valueOptions = getValueOptions();
		const index_t spatialDims = getSpatialDims();
		const index_t boundAxis = bound>>1;
		I4 boundarySize = getSizes();
		boundarySize.a[boundAxis] = 1;
		
		torch::Tensor boundaryVelocity = CreateTensor(1, spatialDims, boundarySize, spatialDims, valueOptions);
		torch::Tensor boundaryScalar = CreateTensor(1, getPassiveScalarChannels(), boundarySize, spatialDims, valueOptions);
		
		std::shared_ptr<VaryingDirichletBoundary> vdb = std::make_shared<VaryingDirichletBoundary>(boundarySlip, boundaryVelocity, boundaryScalar, getParentDomain());
		
		torch::Tensor boundaryTransform = GetFaceTransformBoundarySlice(bound);
		vdb->setTransform(boundaryTransform);
		
		setBoundary(bound, vdb);
	} else {
		torch::Tensor boundaryVelocity = torch::zeros({getSpatialDims()}, CPUValueOptions);
		torch::Tensor boundaryScalar = torch::zeros({getPassiveScalarChannels()}, CPUValueOptions);
		
		std::shared_ptr<StaticDirichletBoundary> sdb = std::make_shared<StaticDirichletBoundary>(boundarySlip, boundaryVelocity, boundaryScalar, getParentDomain());
		setBoundary(bound, sdb);
	}
}
/*
void Block::MakeFixedBoundary(const index_t face,
		optional<torch::Tensor> velocity, const BoundaryConditionType velocityType,
		optional<torch::Tensor> passiveScalar, optional<BoundaryConditionType> scalarType) {
	MakeFixedBoundary(face, velocity, velocityType, passiveScalar, scalarType ? {scalarType} : nullopt);
}*/

void Block::MakeFixedBoundary(const index_t face,
		optional<torch::Tensor> velocity, const BoundaryConditionType velocityType,
		optional<torch::Tensor> passiveScalar, optional<std::vector<BoundaryConditionType>> scalarType) {
	CheckFaceIndex(face);
	
	if(velocity || passiveScalar || hasFaceTransform()){
		optional<torch::Tensor> boundaryTransform = nullopt;
		if(hasFaceTransform()){
			boundaryTransform = GetFaceTransformBoundarySlice(face);
		}
		std::shared_ptr<FixedBoundary> fb = std::make_shared<FixedBoundary>(velocity, velocityType, passiveScalar, nullopt, boundaryTransform, getParentDomain());
		if(scalarType) fb->setPassiveScalarType(scalarType.value());
		setBoundary(face, fb);
	} else {
		torch::Tensor boundaryVelocity = torch::zeros({1, getSpatialDims()}, getValueOptions()); // static (NC), needed to indicate dimensionality
		std::shared_ptr<FixedBoundary> fb = std::make_shared<FixedBoundary>(boundaryVelocity, velocityType, nullopt, nullopt, nullopt, getParentDomain());
		if(scalarType) fb->setPassiveScalarType(scalarType.value());
		setBoundary(face, fb);
	}
}

void Block::ConnectBlock(const dim_t face1, std::shared_ptr<Block> block2, const dim_t face2, const dim_t connectedAxis1, const dim_t connectedAxis2){
	// TODO: check if connection already exists
	CheckFaceIndex(face1);
	CloseConnectedBoudary(face1, true);
	block2->CheckFaceIndex(face2);
	block2->CloseConnectedBoudary(face2, true);
	ConnectBlocks(shared_from_this(), face1, block2, face2, connectedAxis1, connectedAxis2);
}
void Block::ConnectBlock(const std::string &face1, std::shared_ptr<Block> block2, const std::string &face2, const std::string &connectedAxis1, const std::string &connectedAxis2){
	ConnectBlock(BoundarySideToIndex(face1), block2, BoundarySideToIndex(face2), BoundarySideToIndex(connectedAxis1), BoundarySideToIndex(connectedAxis2));
}

void Block::MakePeriodic(const index_t axis){
	CheckAxisIndex(axis);
	const index_t faceLower = axis << 1;
	const index_t faceUpper = faceLower | 1;
	std::shared_ptr<Boundary> boundaryLower = getBoundary(faceLower);
	std::shared_ptr<Boundary> boundaryUpper = getBoundary(faceUpper);
	if(!(boundaryLower->type==BoundaryType::PERIODIC && boundaryUpper->type==BoundaryType::PERIODIC)){
		if(boundaryLower->type==BoundaryType::CONNECTED_GRID){ CloseConnectedBoudary(faceLower, true); }
		if(boundaryUpper->type==BoundaryType::CONNECTED_GRID){ CloseConnectedBoudary(faceUpper, true); }
		
		if(boundaryLower->type!=BoundaryType::PERIODIC){
			std::shared_ptr<PeriodicBoundary> pbLower = std::make_shared<PeriodicBoundary>(getParentDomain());
			setBoundary(faceLower, pbLower);
		}
		if(boundaryUpper->type!=BoundaryType::PERIODIC){
			std::shared_ptr<PeriodicBoundary> pbLower = std::make_shared<PeriodicBoundary>(getParentDomain());
			setBoundary(faceUpper, pbLower);
		}
	}
}
void Block::MakePeriodic(const std::string &axis){
	MakePeriodic(AxisToIndex(axis));
}
void Block::MakeAllPeriodic() {
	for(index_t dim=0; dim<getSpatialDims(); ++dim){
		MakePeriodic(dim);
	}
}

void Block::CloseBoundary(const index_t bound){
	CheckFaceIndex(bound);
	CloseConnectedBoudary(bound, false);
	MakeClosedBoundary(bound);
}
void Block::CloseBoundary(const std::string &face){
	CloseBoundary(BoundarySideToIndex(face));
}
void Block::CloseAllBoundaries(){
	for(index_t face=0; face<getSpatialDims()*2; ++face){
		CloseBoundary(face, nullopt, nullopt);
	}
}

void Block::CloseBoundary(const index_t bound, optional<torch::Tensor> velocity, optional<torch::Tensor> passiveScalar) {
	CheckFaceIndex(bound);
	CloseConnectedBoudary(bound, true);
	MakeFixedBoundary(bound, velocity, BoundaryConditionType::DIRICHLET, passiveScalar, nullopt);
}
void Block::CloseBoundary(const std::string & face, optional<torch::Tensor> velocity, optional<torch::Tensor> passiveScalar) {
	CloseBoundary(BoundarySideToIndex(face), velocity, passiveScalar);
}

bool Block::IsUnconnectedBoundary(const index_t index) const {
    BoundaryType bt = boundaries.at(index)->type;
	return bt==BoundaryType::FIXED || bt==BoundaryType::DIRICHLET || bt==BoundaryType::DIRICHLET_VARYING || bt==BoundaryType::NEUMANN;
}
bool Block::hasPrescribedBoundary() const {
	for(auto bound : boundaries){
		BoundaryType bt = bound->type;
		if(bt==BoundaryType::FIXED || bt==BoundaryType::DIRICHLET || bt==BoundaryType::DIRICHLET_VARYING || bt==BoundaryType::NEUMANN){
			return true;
		}
	}
	return false;
}
void Block::setVertexCoordinates(torch::Tensor &newVertexCoordinates){
	CHECK_INPUT_CUDA(newVertexCoordinates);
	TORCH_CHECK(newVertexCoordinates.dim()==velocity.dim(), "new vertex coordinates dimensions must match velocity.");
	TORCH_CHECK(newVertexCoordinates.size(0)==1, "batches are not yet supported.");
	TORCH_CHECK(newVertexCoordinates.size(1)==getSpatialDims(), "vertex coordinates channel dimension must match spatial dimensions.");
	for(int dim=2;dim<velocity.dim();++dim){
		TORCH_CHECK(velocity.size(dim)==(newVertexCoordinates.size(dim)-1), "vertex coordinates spatial dimension must match velocity's +1.");
	}
	TORCH_CHECK(newVertexCoordinates.dtype()==velocity.dtype(), "Vertex coordinates and velocity must have same dtype.");
	
	clearCoordsTransforms();
	m_vertexCoordinates = newVertexCoordinates;
	m_transform = CoordsToTransforms(newVertexCoordinates);
	m_faceTransform = CoordsToFaceTransforms(newVertexCoordinates);
	
	UpdateBoundaryTransforms();
	
	isTensorChanged = true;
}
void Block::setTransform(torch::Tensor &newTransform, optional<torch::Tensor> newFaceTransform) {
	CHECK_INPUT_CUDA(newTransform);
	TORCH_CHECK(newTransform.dim()==velocity.dim(), "new ransform dimensions must match velocity.");
	TORCH_CHECK(newTransform.size(0)==1, "batches are not yet supported.");
	TORCH_CHECK(newTransform.size(-1)==TransformNumValues(getSpatialDims()), "Transform channels must match spatial dimensions");
	for(int dim=2;dim<velocity.dim();++dim){
		TORCH_CHECK(velocity.size(dim)==newTransform.size(dim-1), "Transform spatial dimension must match velocity's");
	}
	TORCH_CHECK(newTransform.dtype()==velocity.dtype(), "Transform and velocity must have same dtype.");
	
	if(newFaceTransform){
		torch::Tensor faceTransform = newFaceTransform.value();
		CHECK_INPUT_CUDA(faceTransform);
		TORCH_CHECK(faceTransform.dim()==(velocity.dim()+1), "new face transform dimensions must match velocity.");
		TORCH_CHECK(faceTransform.size(0)==1, "batches are not yet supported.");
		TORCH_CHECK(faceTransform.size(1)==getSpatialDims(), "channel dimension must match spatial dimensions.");
		TORCH_CHECK(faceTransform.size(-1)==TransformNumValues(getSpatialDims()), "Transform data must match spatial dimensions.");
		for(int dim=2;dim<velocity.dim();++dim){
			TORCH_CHECK(velocity.size(dim)==(faceTransform.size(dim)-1), "Face transform spatial dimension must match velocity's +1.");
		}
		TORCH_CHECK(faceTransform.dtype()==velocity.dtype(), "Face transform and velocity must have same dtype.");
	}
	
	clearCoordsTransforms();
	m_transform = newTransform;
	m_faceTransform = newFaceTransform;
	
	UpdateBoundaryTransforms();
	
	isTensorChanged = true;
}
/*
void Block::setFaceTransform(torch::Tensor &newTransform) {
	TORCH_CHECK(m_transformType==TransformType::TRANSFORM, "cell tranform has to be set before face transform.");
	CHECK_INPUT_CUDA(newTransform);
	TORCH_CHECK(newTransform.dim()==(velocity.dim()+1), "new Transform dimensions must match velocity.");
	TORCH_CHECK(newTransform.size(0)==1, "batches are not yet supported.");
	TORCH_CHECK(newTransform.size(1)==getSpatialDims(), "channel dimension must match spatial dimensions.");
	TORCH_CHECK(newTransform.size(-1)==TransformNumValues(getSpatialDims()), "Transform data must match spatial dimensions.");
	for(int dim=2;dim<velocity.dim();++dim){
		TORCH_CHECK(velocity.size(dim)==(newTransform.size(dim)-1), "spatial dimension must match +1.");
	}
	TORCH_CHECK(newTransform.dtype()==velocity.dtype(), "Transform and velocity must have same dtype.");
	
	m_faceTransform = newTransform;
	isTensorChanged = true;
}*/
void Block::clearCoordsTransforms(){
	isTensorChanged = isTensorChanged || hasTransform() || hasFaceTransform() || hasVertexCoordinates();
	m_transform = nullopt;
	m_faceTransform = nullopt;
	m_vertexCoordinates = nullopt;
}
torch::Tensor Block::getCellCoordinates() const{
	TORCH_CHECK(hasVertexCoordinates(), "VertexCoordinates are required to compute cell coordinates.")
	switch(getSpatialDims()){
		case 1:
			return torch::avg_pool1d(m_vertexCoordinates.value(), {2}, {1}); // data, kernel size, stride
		case 2:
			return torch::avg_pool2d(m_vertexCoordinates.value(), {2,2}, {1,1});
		case 3:
			return torch::avg_pool3d(m_vertexCoordinates.value(), {2,2,2}, {1,1,1});
		default:
			TORCH_CHECK(false, "only 1-3D is supported.");
			return torch::empty(0);
	}
	
}

index_t Block::GetCoordinateOrientation() const {
	if(!hasTransform()){
		return 1;
	}
	
	torch::Tensor det_sign = torch::sign(m_transform.value().index({torch::indexing::Ellipsis, -1}));
	const index_t sign_min = torch::min(det_sign).cpu().to(torch_kIndex).data_ptr<index_t>()[0];
	const index_t sign_max = torch::max(det_sign).cpu().to(torch_kIndex).data_ptr<index_t>()[0];
	
	if(sign_min!=sign_max){
		return 0;
	}else{
		return sign_min;
	}
}

index_t Block::getSpatialDims() const {
    //return numDims - 2;
	return getParentDomain()->getSpatialDims();
}

index_t Block::getDim(const dim_t dim) const {
    return velocity.size(dim);
}

index_t Block::getAxis(const dim_t dim) const {
	TORCH_CHECK(0<=dim && dim<getSpatialDims(), "Axis index must be within spatial dimensions");
    return velocity.size(velocity.dim()-1 -dim);
}


I4 Block::getSizes() const {
	switch (getSpatialDims())
	{
	case 1:
		return {{.x=getDim(2), .y=1, .z=1, .w=1}};
		break;
	case 2:
		return {{.x=getDim(3), .y=getDim(2), .z=1, .w=2}};
		break;
	case 3:
		return {{.x=getDim(4), .y=getDim(3), .z=getDim(2), .w=3}};
		break;
	
	default:
		return {{.x=1, .y=1, .z=1, .w=1}};
	}
}

I4 Block::getStrides() const {
    const I4 sizes = getSizes();
	return {{.x=1, .y=sizes.x, .z=sizes.x*sizes.y, .w=sizes.x*sizes.y*sizes.z}};
}

index_t Block::ComputeCSRSize() const {
	const I4 size = getSizes();
	const I4 stride = getStrides();
    index_t csrSize = stride.w*(2*getSpatialDims()+1);
	for(dim_t dim = 0; dim<getSpatialDims(); ++dim){
		index_t dimBoundArea = size.a[(dim+1)%3] * size.a[(dim+2)%3];
		if(IsUnconnectedBoundary(dim*2)) csrSize -= dimBoundArea;
		if(IsUnconnectedBoundary(dim*2+1)) csrSize -= dimBoundArea;
	}
    return csrSize;
};

bool Block::IsTensorChanged() const {
	if(isTensorChanged) { return true; }
	for(auto boundary : boundaries){
		if(boundary->IsTensorChanged()){ return true; }
	}
	return false;
}
void Block::setTensorChanged(const bool changed){
	isTensorChanged = changed;
	for(auto boundary : boundaries){
		boundary->setTensorChanged(changed);
	}
}

std::string Block::ToString() const {
	
	std::ostringstream repr;
	I4 size = getSizes();
	I4 stride = getStrides();
	repr << "Block[\"" << name << "\" " << getSpatialDims() << "D";
	repr << ", scalarChannels=" << getPassiveScalarChannels();
	repr << ", transforms=(";
	if(m_vertexCoordinates || m_transform || m_faceTransform){
		if(m_vertexCoordinates) { repr << "C,"; }
		if(m_transform) { repr << "T,"; }
		if(m_faceTransform) { repr << "F,"; }
	} else {
		repr << "none";
	}
	repr << ")";
	//repr << ", dims=" << static_cast<index_t>(numDims) << ", spatial=" << getSpatialDims();
	repr << ", size=" << I4toString(size) << ", stride=" << I4toString(stride);
	repr << ", bounds=( ";
	const index_t numBounds = static_cast<index_t>(boundaries.size());
	for(index_t boundIdx=0; boundIdx<numBounds; ++boundIdx){
		repr << BoundaryIndexToString(boundIdx) << "=" << boundaries[boundIdx]->ToString() << " ";
	}
	repr << ")]";
	return repr.str();
}

// --- Domain ---

//Domain::Domain(std::string &name) : name(name), initialized(false) {}
Domain::Domain(const index_t spatialDims, torch::Tensor &viscosity,
		const std::string &name, const torch::Dtype dtype, optional<py::object> pyDtype, const torch::Device device,
		const index_t passiveScalarChannels, optional<torch::Tensor> passiveScalarViscosity)
		: name(name), pyDtype(pyDtype), m_spatialDims(spatialDims), m_passiveScalarChannels(passiveScalarChannels),
		m_dtype(dtype), m_device(device), initialized(false) {
	//
	TORCH_CHECK(0<m_spatialDims && m_spatialDims<4, "spatialDims must be 1, 2, or 3.");
	TORCH_CHECK(m_passiveScalarChannels>=0, "passiveScalarChannels can't be negative.");
	//TORCH_CHECK(m_passiveScalarChannels==1, "Currently only 1 passiveScalarChannel is supported.");

	setViscosity(viscosity);

	/*TORCH_CHECK(!passiveScalarViscosity.has_value(), "passiveScalarViscosity is not yet supported.");
	if(passiveScalarViscosity){
		CHECK_INPUT_CUDA(passiveScalarViscosity.value());
		TORCH_CHECK(passiveScalarViscosity.value().dim()==1, "passiveScalarViscosity must be 1D.");
		TORCH_CHECK(passiveScalarViscosity.value().size(0)==m_passiveScalarChannels, "passiveScalarViscosity size must match passiveScalarChannels.");
	}*/
	if(passiveScalarViscosity){
		setScalarViscosity(passiveScalarViscosity.value());
	}

	TORCH_CHECK(m_device.is_cuda(), "device must be a CUDA device.");
	if(!m_device.has_index()){ m_device.set_index(0); }
	
	valueOptions = torch::TensorOptions().dtype(dtype).layout(torch::kStrided).device(device.type(), device.index());
}
#ifdef DTOR_MSG
Domain::~Domain(){
	py::print("Domain dtor");
	//DetachFwd();
}
#endif

std::shared_ptr<Domain> Domain::Copy(optional<std::string> newName) const{
	std::string n = newName.value_or(name+"_copy");
	torch::Tensor v = viscosity;
	std::shared_ptr<Domain> cDomain = std::make_shared<Domain>(getSpatialDims(), v, n, getDtype(), pyDtype, getDevice(), getPassiveScalarChannels(), passiveScalarViscosity);
	
	for(auto block : blocks){
		cDomain->AddBlock(block->Copy());
	}
	
	// copy connected boundaries
	for(index_t blockIdx=0, numBlocks=blocks.size(); blockIdx<numBlocks; ++blockIdx){
		for(index_t boundIdx=0, numBounds=blocks[blockIdx]->boundaries.size(); boundIdx<numBounds; ++boundIdx){
			if(blocks[blockIdx]->boundaries[boundIdx]->type==BoundaryType::CONNECTED_GRID){
				std::shared_ptr<ConnectedBoundary> boundary = std::static_pointer_cast<ConnectedBoundary> (blocks[blockIdx]->boundaries[boundIdx]);
				index_t connectedIdx = getBlockIdx(boundary->getConnectedBlock());
				std::shared_ptr<ConnectedBoundary> cBoundary = std::make_shared<ConnectedBoundary>(cDomain->getBlock(connectedIdx), boundary->axes, shared_from_this());
				cDomain->getBlock(blockIdx)->setBoundary(boundIdx, cBoundary);
			}
		}
	}
	
	return cDomain;
}
std::shared_ptr<Domain> Domain::Clone(optional<std::string> newName) const{
	std::string n = newName.value_or(name+"_clone");
	torch::Tensor v = viscosity.clone();
	optional<torch::Tensor> psv = cloneOptionalTensor(passiveScalarViscosity);
	std::shared_ptr<Domain> cDomain = std::make_shared<Domain>(getSpatialDims(), v, n, getDtype(), pyDtype, getDevice(), getPassiveScalarChannels(), psv);
	
	for(auto block : blocks){
		cDomain->AddBlock(block->Clone());
	}
	
	// copy connected boundaries
	for(index_t blockIdx=0, numBlocks=blocks.size(); blockIdx<numBlocks; ++blockIdx){
		for(index_t boundIdx=0, numBounds=blocks[blockIdx]->boundaries.size(); boundIdx<numBounds; ++boundIdx){
			if(blocks[blockIdx]->boundaries[boundIdx]->type==BoundaryType::CONNECTED_GRID){
				std::shared_ptr<ConnectedBoundary> boundary = std::static_pointer_cast<ConnectedBoundary> (blocks[blockIdx]->boundaries[boundIdx]);
				index_t connectedIdx = getBlockIdx(boundary->getConnectedBlock());
				std::shared_ptr<ConnectedBoundary> cBoundary = std::make_shared<ConnectedBoundary>(cDomain->getBlock(connectedIdx), boundary->axes, shared_from_this());
				cDomain->getBlock(blockIdx)->setBoundary(boundIdx, cBoundary);
			}
		}
	}
	
	return cDomain;
}
std::shared_ptr<Domain> Domain::To(const torch::Dtype dtype, optional<std::string> newName){
	if(dtype==m_dtype){
		return shared_from_this();
	} else {
		TORCH_CHECK(false, "To(dtype) not implemented.");
		return shared_from_this();
	}
}
std::shared_ptr<Domain> Domain::To(const torch::Device device, optional<std::string> newName){
	if(device==getDevice()){
		return shared_from_this();
	} else {
		TORCH_CHECK(device.is_cuda(), "device must be a CUDA device.");
		TORCH_CHECK(false, "To(device) not implemented.");
		return shared_from_this();
	}
}

void Domain::AddBlock(std::shared_ptr<Block> block){
	TORCH_CHECK(block->getSpatialDims()==getSpatialDims(), "Number of block spatial dimensions does not match domain.");
	TORCH_CHECK(block->getPassiveScalarChannels()==getPassiveScalarChannels(), "Number of block passive scalar channels does not match domain.");
	TORCH_CHECK(block->getDtype()==getDtype(), "Data typ does not match.");
	TORCH_CHECK(block->getDevice()==getDevice(), "Device does not match.");

	blocks.push_back(block);
	initialized = false;
}


std::shared_ptr<Block> Domain::CreateBlock( optional<torch::Tensor> velocity, optional<torch::Tensor> pressure, optional<torch::Tensor> passiveScalar,
		optional<torch::Tensor> vertexCoordinates, const std::string &name){
	/*CHECK_INPUT_CUDA(velocity);
	TORCH_CHECK(velocity.dim()==m_spatialDims+2, "velocity must be " + std::to_string(m_spatialDims) + "D.");
	TORCH_CHECK(velocity.size(0)==1, "Batches (dim 0) are not yet supported (must be 1).");
	TORCH_CHECK(velocity.size(1)==m_spatialDims, "Channels (dim 1) of velocity must match spatial dimensions ()" + std::to_string(m_spatialDims) + ").");
	TORCH_CHECK(velocity.dtype()==m_dtype, "velocity has wrong dtype.");
	TORCH_CHECK(velocity.device()==m_device, "velocity has wrong device.");

	TORCH_CHECK(passiveScalar.has_value()==hasPassiveScalar(), "");
	if(passiveScalar){
		TORCH_CHECK(passiveScalar.value().size(1)==m_passiveScalarChannels, "passiveScalar channels do not match domain.");
	}*/
	
	std::shared_ptr<Block> p_block = std::make_shared<Block>(velocity, pressure, passiveScalar, vertexCoordinates, name, shared_from_this());
	
	TORCH_CHECK(p_block->getDevice()==getDevice(), "Block has wrong device");
	//TORCH_CHECK(CompareDevice(p_block->getDevice(), getDevice()), "Block has wrong device");
	//TORCH_CHECK(p_block->getDevice().type()==getDevice().type(), "Block has wrong device type");
	//TORCH_CHECK(p_block->getDevice().index()==getDevice().index(), "Block has wrong device index: Block has "
	//	+ std::to_string(p_block->getDevice().index()) + ", domain expects " + std::to_string(getDevice().index()));
	//TORCH_CHECK(p_block->getDtype()==getDtype(), "Block has wrong dtype");
	//TORCH_CHECK(p_block->getSpatialDims()==getSpatialDims(), "Block must be " + std::to_string(m_spatialDims) + "D.");
	//TORCH_CHECK(p_block->getPassiveScalarChannels()==getPassiveScalarChannels(), "passiveScalar channels do not match domain.");
	
	AddBlock(p_block);
	
	return p_block;
}

std::shared_ptr<Block> Domain::CreateBlockWithSize(I4 size, std::string &name){
	TORCH_CHECK(size.x>2, "size.x must be at least 3.");
	if(getSpatialDims()>1) TORCH_CHECK(size.y>2, "size.y must be at least 3.");
	if(getSpatialDims()>2) TORCH_CHECK(size.z>2, "size.z must be at least 3.");
	size.w = getSpatialDims();

	std::shared_ptr<Block> p_block = std::make_shared<Block>(size, name, shared_from_this());
	if(hasPassiveScalar()){
		p_block->CreatePassiveScalar();
	}
	
	AddBlock(p_block);

	return p_block;
}

/*
std::shared_ptr<Block> Domain::CreateBlockFromCoords(torch::Tensor vertexCoordinates, const std::string &name){
	CHECK_INPUT_CUDA(vertexCoordinates);
	TORCH_CHECK(vertexCoordinates.dim()==m_spatialDims+2, "vertexCoordinates must be " + std::to_string(m_spatialDims) + "D.");
	TORCH_CHECK(vertexCoordinates.size(0)==1, "Batches (dim 0) are not yet supported (must be 1).");
	TORCH_CHECK(vertexCoordinates.size(1)==m_spatialDims, "Channels (dim 1) of vertexCoordinates must match spatial dimensions ()" + std::to_string(m_spatialDims) + ").");
	TORCH_CHECK(vertexCoordinates.dtype()==m_dtype, "vertexCoordinates has wrong dtype.");
	TORCH_CHECK(vertexCoordinates.device()==m_device, "vertexCoordinates has wrong device.");
	
	I4 size = {.a={0}};
	size.w = m_spatialDims;
	for(index_t dim=0; dim<m_spatialDims; ++dim){
		size.a[dim] = vertexCoordinates.size(m_spatialDims + 1 - dim) - 1;
	}

	std::shared_ptr<Block> p_block = CreateBlockWithSize(size, name);
	
	p_block->setVertexCoordinates(vertexCoordinates);
	
	AddBlock(p_block);

	return p_block;
}*/

torch::Tensor Domain::getMaxVelocity(const bool withBounds, const bool computational) const{
	TORCH_CHECK(getNumBlocks()>0, "Domain does not contain any blocks.");
	
	torch::Tensor maxVel = blocks[0]->getMaxVelocity(withBounds, computational);
	for(size_t blockIdx = 1; blockIdx < blocks.size(); ++blockIdx){
		maxVel = torch::maximum(maxVel, blocks[blockIdx]->getMaxVelocity(withBounds, computational));
	}
	
	return maxVel;
}

torch::Tensor Domain::getMaxVelocityMagnitude(const bool withBounds, const bool computational) const{
	TORCH_CHECK(getNumBlocks()>0, "Domain does not contain any blocks.");
	
	torch::Tensor maxMag = blocks[0]->getMaxVelocityMagnitude(withBounds, computational);
	for(size_t blockIdx = 1; blockIdx < blocks.size(); ++blockIdx){
		maxMag = torch::maximum(maxMag, blocks[blockIdx]->getMaxVelocityMagnitude(withBounds, computational));
	}
	
	return maxMag;
}
bool Domain::hasVertexCoordinates() const {
	for(auto block : blocks){
		if(!block->hasVertexCoordinates()){
			return false;
		}
	}
	
	return getNumBlocks()>0;
}
std::vector<torch::Tensor> Domain::getVertexCoordinates() const {
	TORCH_CHECK(getNumBlocks()>0, "Domain does not contain any blocks.");
	
	std::vector<torch::Tensor> coordList;
	for(auto block : blocks){
		TORCH_CHECK(block->hasVertexCoordinates(), "a block is missing vertex coordinates");
		coordList.push_back(block->m_vertexCoordinates.value());
	}
	
	return coordList;
}

index_t Domain::GetCoordinateOrientation() const {
	TORCH_CHECK(getNumBlocks()>0, "Domain does not contain any blocks.");
	
	index_t sign = blocks[0]->GetCoordinateOrientation();
	for(index_t i=1; i<getNumBlocks(); ++i){
		if(blocks[i]->GetCoordinateOrientation() != sign){
			return 0;
		}
	}
	
	return sign;
}

bool Domain::hasPrescribedBoundary() const {
	for(const auto &block : blocks){
		if(block->hasPrescribedBoundary()){
			return true;
		}
	}
	return false;
}

bool Domain::isAllFixedBoundariesPassiveScalarTypeStatic() const{
	for(const auto &block : blocks){
		if(!block->isAllFixedBoundariesPassiveScalarTypeStatic()){
			return false;
		}
	}
	return true;
}

torch::Tensor Domain::GetGlobalFluxBalance() const {
	torch::Tensor fluxSum = torch::zeros({1}, valueOptions);
	torch::Tensor mOne = torch::tensor({-1}, valueOptions);
	
	for(auto block : blocks){
		// std::shared_ptr<Block> block;
		index_t boundIdx = 0;
		for(auto boundary : block->getBoundaries()){
			switch(boundary->type){
				case BoundaryType::FIXED: {
					std::shared_ptr<FixedBoundary> bound = std::static_pointer_cast<FixedBoundary>(boundary);
					torch::Tensor boundaryFlux = torch::sum(bound->GetFluxes());
					
					if(!(boundIdx&1)){ // is lower boundary
						boundaryFlux = boundaryFlux * mOne;
					}
					
					fluxSum = fluxSum + boundaryFlux;
					break;
				}
				case BoundaryType::DIRICHLET:
				case BoundaryType::DIRICHLET_VARYING:
				case BoundaryType::NEUMANN:{
					TORCH_CHECK(false, "Old boundaries not supported.");
					break;
				}
				default:
					break;
			}
			++boundIdx;
		}
	}
	
	return fluxSum;
}
bool Domain::CheckGlobalFluxBalance(const double eps) const {
	const double fluxBalance = GetGlobalFluxBalance().cpu().to(torch::kFloat64).data_ptr<double>()[0];
	return abs(fluxBalance)<eps;
}

void Domain::setViscosity(torch::Tensor &viscosity){
	CHECK_INPUT_HOST(viscosity);
	TORCH_CHECK(viscosity.dim()==1, "viscosity must be 1D.");
	TORCH_CHECK(viscosity.size(0)==1, "viscosity must be a scalar.");
	
	this->viscosity = viscosity;
	if(initialized){
		AT_DISPATCH_FLOATING_TYPES(getDtype(), "SetupDomainGPU", ([&] {
			SetViscosityGPU<scalar_t>();
		}));
	}
}

bool Domain::hasBlockViscosity() const {
	for(const auto &block : blocks){
		if(block->hasViscosity()){
			return true;
		}
	}
	return false;
}

template <typename scalar_t>
void Domain::SetViscosityGPU(){
	DomainGPU<scalar_t> *p_host_domain= reinterpret_cast<DomainGPU<scalar_t>*>(atlas.p_host);
	DomainGPU<scalar_t> *p_device_domain= reinterpret_cast<DomainGPU<scalar_t>*>(atlas.p_device);
	p_host_domain->viscosity = viscosity.data_ptr<scalar_t>()[0];
	CopyToGPU(&p_device_domain->viscosity, &p_host_domain->viscosity, sizeof(scalar_t));

}
void Domain::setScalarViscosity(torch::Tensor &viscosity){
	TORCH_CHECK(viscosity.dim()==1, "Scalar viscosity must be 1D.");
	TORCH_CHECK(viscosity.size(0)==1 || viscosity.size(0)==m_passiveScalarChannels, "Scalar viscosity must be static (a scalar) or match passive scalar channels.");
	TORCH_CHECK(viscosity.scalar_type()==getDtype(), "Data type of scalar viscosity does not match.");
	CHECK_INPUT_CUDA(viscosity);
	//TORCH_CHECK(viscosity.device()==getDevice(), "Device of scalar viscosity does not match.");

	passiveScalarViscosity = viscosity;
	isTensorChanged=true;
}
void Domain::clearScalarViscosity() {
	
	passiveScalarViscosity = nullopt;
	isTensorChanged=true;
}
bool Domain::hasPassiveScalarBlockViscosity() const {
	for(const auto &block : blocks){
		if(block->hasPassiveScalarViscosity()){
			return true;
		}
	}
	return false;
}

void Domain::PrepareSolve(){
	//if(initialized) return;
	TORCH_CHECK(getNumBlocks()>0, "Domain does not contain any blocks.");
	TORCH_CHECK(viscosity.scalar_type()==getDtype(), "Data type of viscosity does not match.");
	
	//const bool hasPassiveScalar = blocks[0]->hasPassiveScalar();
	//const index_t passiveScalarChannels = blocks[0]->getPassiveScalarChannels();
	
	index_t csrSize = 0;
	totalSize = 0;
	for(auto block : blocks){
		block->csrOffset = csrSize;
		block->globalOffset = totalSize;
		
		csrSize += block->ComputeCSRSize();
		totalSize += block->getStrides().w;
		
		
		//for(const auto boundary : block->boundaries){
		const index_t numBounds = static_cast<index_t>(block->boundaries.size());
		for(index_t boundIdx=0; boundIdx<numBounds; ++boundIdx){
			std::shared_ptr<Boundary> boundary = block->boundaries[boundIdx];
			switch (boundary->type)
			{
			case BoundaryType::DIRICHLET:
			case BoundaryType::DIRICHLET_VARYING: {
				py::print("Warning: Dirichlet(Varying/Static) boundaries are deprecated, use FixedBoundary instead.");
			}
			/*case BoundaryType::FIXED: {
				std::shared_ptr<FixedBoundary> bound = std::static_pointer_cast<FixedBoundary> (boundary);
				break;
			}*/
			case BoundaryType::CONNECTED_GRID: {
				//std::shared_ptr<ConnectedBoundary> bound = std::dynamic_pointer_cast<ConnectedBoundary> (boundary);
				std::shared_ptr<ConnectedBoundary> bound = std::static_pointer_cast<ConnectedBoundary> (boundary);
				std::shared_ptr<Block> otherBlock = bound->getConnectedBlock();
				TORCH_CHECK(getBlockIdx(otherBlock)>=0, "Connected block is not part of this domain.");
				//check that the connection goes both ways
				const index_t otherBoundIdx = bound->axes[0];
				std::shared_ptr<Boundary> otherBoundary = otherBlock->getBoundary(otherBoundIdx);
				TORCH_CHECK(otherBoundary->type==BoundaryType::CONNECTED_GRID, "Mismatch in block connection: connected block is not connected at the target face.");
				std::shared_ptr<ConnectedBoundary> otherBound = std::static_pointer_cast<ConnectedBoundary> (otherBoundary);
				TORCH_CHECK(block==otherBound->getConnectedBlock(), "Mismatch in block connection: connected block is connected to a different block at the target face.");
				TORCH_CHECK(boundIdx==otherBound->axes[0], "Mismatch in block connection: connected block is connected to a different face of the source block.");
				//TODO: check axis alignments
				// axis sizes are check on adding the boundary to a block
				const dim_t spatialDims = getSpatialDims();
				if(spatialDims==2){
					TORCH_CHECK(bound->getConnectionAxisDirection(1)==otherBound->getConnectionAxisDirection(1), "Direction of connection does not match.");
				}
				if(spatialDims==3){
					//valid configs:
					if(bound->getConnectionAxis(1) == (bound->getConnectedFace() + 1)%spatialDims){
						// axis order not swapped
						TORCH_CHECK(bound->getConnectionAxis(2) == (bound->getConnectedFace() + 2)%spatialDims, "Invalid connection configuration: axis2 has wrong target axis.");
						TORCH_CHECK(otherBound->getConnectionAxis(1) == (otherBound->getConnectedFace() + 1)%spatialDims, "Invalid connection configuration: axis1 has wrong target axis.");
						TORCH_CHECK(otherBound->getConnectionAxis(2) == (otherBound->getConnectedFace() + 2)%spatialDims, "Invalid connection configuration: axis2 has wrong target axis.");
						TORCH_CHECK(bound->getConnectionAxisDirection(1) == otherBound->getConnectionAxisDirection(1), "Invalid connection configuration: Direction of connection axis1-axis1 does not match.");
						TORCH_CHECK(bound->getConnectionAxisDirection(2) == otherBound->getConnectionAxisDirection(2), "Invalid connection configuration: Direction of connection axis2-axis2 does not match.");
						
					} else if(bound->getConnectionAxis(1) == (bound->getConnectedFace() + 2)%spatialDims){
						// axis order swapped
						TORCH_CHECK(bound->getConnectionAxis(2) == (bound->getConnectedFace() + 1)%spatialDims, "Invalid connection configuration: axis2 has wrong target axis.");
						TORCH_CHECK(otherBound->getConnectionAxis(1) == (otherBound->getConnectedFace() + 2)%spatialDims, "Invalid connection configuration: axis1 has wrong target axis.");
						TORCH_CHECK(otherBound->getConnectionAxis(2) == (otherBound->getConnectedFace() + 1)%spatialDims, "Invalid connection configuration: axis2 has wrong target axis.");
						TORCH_CHECK(bound->getConnectionAxisDirection(1) == otherBound->getConnectionAxisDirection(2), "Invalid connection configuration: Direction of connection axis1-axis2 does not match.");
						TORCH_CHECK(bound->getConnectionAxisDirection(2) == otherBound->getConnectionAxisDirection(1), "Invalid connection configuration: Direction of connection axis2-axis1 does not match.");
						
					} else {
						TORCH_CHECK(false, "Invalid connection configuration: axis1 has invalid target axis.");
					}
				}
				break;
			}
			case BoundaryType::PERIODIC: {
				TORCH_CHECK(block->boundaries[boundIdx^1]->type==BoundaryType::PERIODIC, "Opposite boundary must also be periodic.");
				break;
			}
			default:
				break;
			}
		}
	}

	//valueOptions = torch::TensorOptions().dtype(getDtype()).layout(torch::kStrided).device(getDevice().type(), getDevice().index());
	//auto indexOptions = torch::TensorOptions().dtype(torch_kIndex).layout(torch::kStrided).device(getDevice().type(), getDevice().index());

	C = std::make_shared<CSRmatrix>(csrSize, totalSize, getDtype(), getDevice());
	P = std::make_shared<CSRmatrix>(csrSize, totalSize, getDtype(), getDevice());
	//A = torch::zeros(totalSize, valueOptions);
	CreateA();
#ifdef WITH_GRAD
	C_grad = C->WithZeroValue(); //shared indexing tensors
	P_grad = P->WithZeroValue(); //shared indexing tensors
#endif
	
	if(hasPassiveScalar()){
		//scalarRHS = torch::zeros(totalSize, valueOptions);
		CreateScalarRHS();
		//scalarResult = torch::zeros(totalSize, valueOptions);
		CreateScalarResult();
	}
	
	//velocityRHS = torch::zeros(totalSize*getSpatialDims(), valueOptions);
	CreateVelocityRHS();
	//velocityResult = torch::zeros(totalSize*getSpatialDims(), valueOptions);
	CreateVelocityResult();
	
	//pressureRHS = torch::zeros(totalSize*getSpatialDims(), valueOptions);
	CreatePressureRHS();
	//pressureRHSdiv = torch::zeros(totalSize, valueOptions);
	CreatePressureRHSdiv();
	//pressureResult = torch::zeros(totalSize, valueOptions);
	CreatePressureResult();
	isTensorChanged=true;
	
	
	AT_DISPATCH_FLOATING_TYPES(getDtype(), "SetupDomainGPU", ([&] {
		SetupDomainGPU<scalar_t>();
	}));

	initialized = true;
}

void Domain::DetachFwd() {
	for(std::shared_ptr<Block> block : blocks){
		block->DetachFwd();
	}
	A = A.detach();
	C->Detach();
	P->Detach();
	if(!IsTensorEmpty(scalarRHS)){ scalarRHS = scalarRHS.detach(); }
	if(!IsTensorEmpty(scalarResult)){ scalarResult = scalarResult.detach(); }
	velocityRHS = velocityRHS.detach();
	velocityResult = velocityResult.detach();
	pressureRHS = pressureRHS.detach();
	pressureRHSdiv = pressureRHSdiv.detach();
	pressureResult = pressureResult.detach();
}
void Domain::DetachGrad() {
	for(std::shared_ptr<Block> block : blocks){
		block->DetachGrad();
	}
	A_grad = A_grad.detach();
	C_grad->Detach();
	P_grad->Detach();
	//P_grad->Detach();
	if(!IsTensorEmpty(scalarRHS_grad)){ scalarRHS_grad = scalarRHS_grad.detach(); }
	if(!IsTensorEmpty(scalarResult_grad)){ scalarResult_grad = scalarResult_grad.detach(); }
	velocityRHS_grad = velocityRHS_grad.detach();
	velocityResult_grad = velocityResult_grad.detach();
	pressureRHS_grad = pressureRHS_grad.detach();
	pressureRHSdiv_grad = pressureRHSdiv_grad.detach();
	pressureResult_grad = pressureResult_grad.detach();
}
void Domain::Detach() {
	DetachFwd();
	DetachGrad();
}
void Domain::CreatePressureOnBlocks(){
	for(auto block : blocks){
		block->CreatePressure();
	}
}
void Domain::CreateVelocityOnBlocks(){
	for(auto block : blocks){
		block->CreateVelocity();
	}
}

void Domain::CreatePassiveScalarOnBlocks(){
	if(hasPassiveScalar()){
		for(auto block : blocks){
			block->CreatePassiveScalar();
		}
	}
}
/* void Domain::clearPassiveScalarOnBlocks(){
	for(auto block : blocks){
		block->clearPassiveScalar();
	}
} */
bool Domain::hasPassiveScalar() const{
	return m_passiveScalarChannels > 0;
}
index_t Domain::getPassiveScalarChannels() const{
	return m_passiveScalarChannels;
}

bool Domain::CheckDataTensor(const torch::Tensor &tensor, const index_t channels, const std::string &name){
	CHECK_INPUT_CUDA(tensor);
	TORCH_CHECK(tensor.dim()==1, "Dimensions of " + name + " must be 1 (flat).");
	TORCH_CHECK(tensor.size(0)==totalSize*channels, "Size of " + name + " must be " + std::to_string(totalSize*channels) + ".");
	TORCH_CHECK(tensor.scalar_type()==getDtype(), name + " has wrong dtype.");
	return true;
}
void Domain::setA(torch::Tensor &a){
	CheckDataTensor(a, 1, "A");
	A = a;
	isTensorChanged=true;
}
void Domain::CreateA(){
	A = torch::zeros(totalSize, valueOptions);
	isTensorChanged=true;
}

void Domain::setScalarRHS(torch::Tensor &srhs){
	TORCH_CHECK(hasPassiveScalar(), "No base passive scalar set.")
	CheckDataTensor(srhs, getPassiveScalarChannels(), "ScalarRHS");
	scalarRHS = srhs;
	isTensorChanged=true;
}
void Domain::CreateScalarRHS(){
	TORCH_CHECK(hasPassiveScalar(), "No base passive scalar set.")
	scalarRHS = torch::zeros(totalSize * getPassiveScalarChannels(), valueOptions);
	isTensorChanged=true;
}
void Domain::setScalarResult(torch::Tensor &sr){
	TORCH_CHECK(hasPassiveScalar(), "No base passive scalar set.")
	CheckDataTensor(sr, getPassiveScalarChannels(), "ScalarResult");
	scalarResult = sr;
	isTensorChanged=true;
}
void Domain::CreateScalarResult(){
	TORCH_CHECK(hasPassiveScalar(), "No base passive scalar set.")
	scalarResult = torch::zeros(totalSize * getPassiveScalarChannels(), valueOptions);
	isTensorChanged=true;
}
void Domain::setVelocityRHS(torch::Tensor &vrhs){
	CheckDataTensor(vrhs, getSpatialDims(), "VelocityRHS");
	velocityRHS = vrhs;
	isTensorChanged=true;
}
void Domain::CreateVelocityRHS(){
	velocityRHS = torch::zeros(totalSize*getSpatialDims(), valueOptions);
	isTensorChanged=true;
}
void Domain::setVelocityResult(torch::Tensor &vr){
	CheckDataTensor(vr, getSpatialDims(), "VelocityResult");
	velocityResult = vr;
	isTensorChanged=true;
}
void Domain::CreateVelocityResult(){
	velocityResult = torch::zeros(totalSize*getSpatialDims(), valueOptions);
	isTensorChanged=true;
}
void Domain::setPressureRHS(torch::Tensor &prhs){
	CheckDataTensor(prhs, getSpatialDims(), "PressureRHS");
	pressureRHS = prhs;
	isTensorChanged=true;
}
void Domain::CreatePressureRHS(){
	pressureRHS = torch::zeros(totalSize*getSpatialDims(), valueOptions);
	isTensorChanged=true;
}
void Domain::setPressureRHSdiv(torch::Tensor &prhsd){
	CheckDataTensor(prhsd, 1, "PressureRHSdiv");
	pressureRHSdiv = prhsd;
	isTensorChanged=true;
}
void Domain::CreatePressureRHSdiv(){
	pressureRHSdiv = torch::zeros(totalSize, valueOptions);
	isTensorChanged=true;
}
void Domain::setPressureResult(torch::Tensor &pr){
	CheckDataTensor(pr, 1, "pressureResult");
	pressureResult = pr;
	isTensorChanged=true;
}
void Domain::CreatePressureResult(){
	pressureResult = torch::zeros(totalSize, valueOptions);
	isTensorChanged=true;
}

#ifdef WITH_GRAD

void Domain::CreatePassiveScalarGradOnBlocks(){
	for(auto block : blocks){
		block->CreatePassiveScalarGrad();
	}
}
void Domain::CreateVelocityGradOnBlocks(){
	for(auto block : blocks){
		block->CreateVelocityGrad();
	}
}
void Domain::CreateVelocitySourceGradOnBlocks(){
	for(auto block : blocks){
		block->CreateVelocitySourceGrad();
	}
}
void Domain::CreatePressureGradOnBlocks(){
	for(auto block : blocks){
		block->CreatePressureGrad();
	}
}

void Domain::CreatePassiveScalarGradOnBoundaries(){
	for(auto block : blocks){
		block->CreatePassiveScalarGradOnBoundaries();
	}
}
void Domain::CreateVelocityGradOnBoundaries(){
	for(auto block : blocks){
		block->CreateVelocityGradOnBoundaries();
	}
}
void Domain::setViscosityGrad(torch::Tensor &t){
	CHECK_INPUT_CUDA(t);
	TORCH_CHECK(t.dim()==1 && t.size(0)==1, "velocity grad must have shape (1).")
	TORCH_CHECK(t.scalar_type()==getDtype(), "velocity grad has wrong dtype.");
	
	viscosity_grad = t;
	isTensorChanged=true;
}
void Domain::clearViscosityGrad(){
	if(hasViscosityGrad()){
		viscosity_grad = nullopt;
		isTensorChanged = true;
	}
}
void Domain::CreateViscosityGrad(){
	torch::Tensor v_grad = torch::zeros({1}, valueOptions);
	setViscosityGrad(v_grad);
	for(auto block : blocks){
		block->CreateViscosityGrad();
	}
	CreatePassiveScalarViscosityGrad();
}

void Domain::setPassiveScalarViscosityGrad(torch::Tensor &t){
	TORCH_CHECK(hasPassiveScalarViscosity(), "Domain has no passive scalar viscosity.");
	TORCH_CHECK(t.dim()==1, "Scalar viscosity grad must be 1D.");
	TORCH_CHECK(t.size(0)==passiveScalarViscosity.value().size(0), "Scalar viscosity grad shape must match Scalar viscosity.");
	TORCH_CHECK(t.scalar_type()==getDtype(), "Data type of scalar viscosity does not match.");
	CHECK_INPUT_CUDA(t);
	
	passiveScalarViscosity_grad = t;
	isTensorChanged = true;
}
void Domain::clearPassiveScalarViscosityGrad(){
	if(hasPassiveScalarViscosityGrad()){
		passiveScalarViscosity_grad = nullopt;
		isTensorChanged = true;
	}
}
void Domain::CreatePassiveScalarViscosityGrad(){
	if(hasPassiveScalarViscosity()){
		torch::Tensor v = torch::zeros_like(passiveScalarViscosity.value());
		setPassiveScalarViscosityGrad(v);
	} else {
		clearPassiveScalarViscosityGrad();
	}
}

void Domain::setAGrad(torch::Tensor &tensor){
	CheckDataTensor(tensor, 1, "AGrad");
	A_grad = tensor;
	isTensorChanged=true;
}
void Domain::CreateAGrad(){
	A_grad = torch::zeros(totalSize, valueOptions);
	isTensorChanged=true;
}

void Domain::setScalarRHSGrad(torch::Tensor &srhsg){
	TORCH_CHECK(hasPassiveScalar(), "No base passive scalar set.")
	CheckDataTensor(srhsg, getPassiveScalarChannels(), "ScalarRHSGrad");
	scalarRHS_grad = srhsg;
	isTensorChanged=true;
}
void Domain::CreateScalarRHSGrad(){
	TORCH_CHECK(hasPassiveScalar(), "No base passive scalar set.")
	scalarRHS_grad = torch::zeros(totalSize*getPassiveScalarChannels(), valueOptions);
	isTensorChanged=true;
}
void Domain::setScalarResultGrad(torch::Tensor &srg){
	TORCH_CHECK(hasPassiveScalar(), "No base passive scalar set.")
	CheckDataTensor(srg, getPassiveScalarChannels(), "ScalarResultGrad");
	scalarResult_grad = srg;
	isTensorChanged=true;
}
void Domain::CreateScalarResultGrad(){
	TORCH_CHECK(hasPassiveScalar(), "No base passive scalar set.")
	scalarResult_grad = torch::zeros(totalSize*getPassiveScalarChannels(), valueOptions);
	isTensorChanged=true;
}

void Domain::setVelocityRHSGrad(torch::Tensor &vrhs){
	CheckDataTensor(vrhs, getSpatialDims(), "velocityRHS_grad");
	velocityRHS_grad = vrhs;
	isTensorChanged=true;
}
void Domain::CreateVelocityRHSGrad(){
	velocityRHS_grad = torch::zeros(totalSize*getSpatialDims(), valueOptions);
	isTensorChanged=true;
}
void Domain::setVelocityResultGrad(torch::Tensor &vr){
	CheckDataTensor(vr, getSpatialDims(), "velocityResult_grad");
	velocityResult_grad = vr;
	isTensorChanged=true;
}
void Domain::CreateVelocityResultGrad(){
	velocityResult_grad = torch::zeros(totalSize*getSpatialDims(), valueOptions);
	isTensorChanged=true;
}

void Domain::setPressureRHSGrad(torch::Tensor &prhs){
	CheckDataTensor(prhs, getSpatialDims(), "pressureRHS_grad");
	pressureRHS_grad = prhs;
	isTensorChanged=true;
}
void Domain::CreatePressureRHSGrad(){
	pressureRHS_grad = torch::zeros(totalSize*getSpatialDims(), valueOptions);
	isTensorChanged=true;
}
void Domain::setPressureRHSdivGrad(torch::Tensor &prhsd){
	CheckDataTensor(prhsd, 1, "pressureRHSdiv_grad");
	pressureRHSdiv_grad = prhsd;
	isTensorChanged=true;
}
void Domain::CreatePressureRHSdivGrad(){
	pressureRHSdiv_grad = torch::zeros(totalSize, valueOptions);
	isTensorChanged=true;
}
void Domain::setPressureResultGrad(torch::Tensor &pr){
	CheckDataTensor(pr, 1, "pressureResult_grad");
	pressureResult_grad = pr;
	isTensorChanged=true;
}
void Domain::CreatePressureResultGrad(){
	pressureResult_grad = torch::zeros(totalSize, valueOptions);
	isTensorChanged=true;
}

#endif //WITH_GRAD

template <typename scalar_t>
void Domain::SetupDomainGPU(){
	
	const size_t domainAlignment = alignof(DomainGPU<scalar_t>); //std::alignment_of<DomainGPU<scalar_t>>::value;
	const size_t blockAlignment =  alignof(BlockGPU<scalar_t>); //std::alignment_of<BlockGPU<scalar_t>>::value;
	const index_t numBlocks = blocks.size();
	const size_t blocksStartOffsetBytes = sizeof(DomainGPU<scalar_t>);
	if(domainAlignment<blockAlignment && blocksStartOffsetBytes%blockAlignment!=0){
		//TODO correct block start offset for alignment
		TORCH_CHECK(false, "Alignment issues.")
	}
	const size_t atlasSizeBytes = blocksStartOffsetBytes + sizeof(BlockGPU<scalar_t>) * numBlocks;
	size_t allocSizeBytes = domainAlignment + atlasSizeBytes;
	
	// py::print("Int4 size:", sizeof(I4));
	// py::print("CSRmatrixGPU size:", sizeof(CSRmatrixGPU<scalar_t>));
	// py::print("BoundaryGPU size:", sizeof(BoundaryGPU<scalar_t>));
	// py::print("DomainGPU size:", sizeof(DomainGPU<scalar_t>), "alignment:", domainAlignment);
	// py::print("BlockGPU size:", sizeof(BlockGPU<scalar_t>), "alignment:", blockAlignment);
	
	auto byteOptions = torch::TensorOptions().dtype(torch::kUInt8).layout(torch::kStrided).device(getDevice().type(), getDevice().index());
	auto byteOptionsCPU = torch::TensorOptions().dtype(torch::kUInt8).layout(torch::kStrided); //.device(getDevice().type(), getDevice().index());
	
	domainCPU = torch::zeros(allocSizeBytes, byteOptionsCPU);
	domainGPU = torch::zeros(allocSizeBytes, byteOptions);
	
	//DomainAtlasSet atlas;
	memset(&atlas, 0, sizeof(DomainAtlasSet));
	atlas.sizeBytes = atlasSizeBytes;
	atlas.blocksOffsetBytes = blocksStartOffsetBytes;
	atlas.p_host = reinterpret_cast<void*>(domainCPU.data_ptr<uint8_t>());
	atlas.p_device = reinterpret_cast<void*>(domainGPU.data_ptr<uint8_t>());
	// py::print("Raw pointer:", allocSizeBytes, "CPU", atlas.p_host, "GPU", atlas.p_device);
	TORCH_CHECK(std::align(domainAlignment, atlasSizeBytes, atlas.p_host, allocSizeBytes), "Failed to align CPU domain.")
	TORCH_CHECK(std::align(domainAlignment, atlasSizeBytes, atlas.p_device, allocSizeBytes), "Failed to align GPU domain.")
	// py::print("Aligned pointer:", allocSizeBytes, "CPU", p_domainCPUdata, "GPU", atlas.p_device);
	
	UpdateDomainGPU<scalar_t>();
}

void Domain::UpdateDomain(){
	if(!initialized) { TORCH_CHECK(false, "Domain is not initialized. Run domain.PrepareSolve() after setup."); }
	if(IsTensorChanged()){
		AT_DISPATCH_FLOATING_TYPES(getDtype(), "UpdateDomainGPU", ([&] {
			UpdateDomainGPU<scalar_t>();
		}));
	}
}
template <typename scalar_t>
void Domain::UpdateDomainGPU(){
	DomainGPU<scalar_t> *p_domainCPU = reinterpret_cast<DomainGPU<scalar_t>*>(atlas.p_host);
	BlockGPU<scalar_t> *p_blocksCPU = reinterpret_cast<BlockGPU<scalar_t>*>(reinterpret_cast<uint8_t*>(atlas.p_host) + atlas.blocksOffsetBytes);
	
	//DomainGPU<scalar_t> *p_domainGPU = reinterpret_cast<DomainGPU<scalar_t>*>(atlas.p_device);
	BlockGPU<scalar_t> *p_blocksGPU = reinterpret_cast<BlockGPU<scalar_t>*>(reinterpret_cast<uint8_t*>(atlas.p_device) + atlas.blocksOffsetBytes);
	
	
	p_domainCPU->numDims = getSpatialDims();
	p_domainCPU->passiveScalarChannels = getPassiveScalarChannels();
	p_domainCPU->numBlocks = blocks.size();
	p_domainCPU->numCells = totalSize;
	p_domainCPU->blocks = p_blocksGPU; //already set correct pointer for GPU version
	
	p_domainCPU->viscosity = viscosity.data_ptr<scalar_t>()[0];
	p_domainCPU->scalarViscosity = hasPassiveScalarViscosity() ? getPassiveScalarViscosityDataPtr<scalar_t>() : nullptr; //viscosity.data_ptr<scalar_t>(); viscosity is a CPU tensor
	p_domainCPU->scalarViscosityStatic = isPassiveScalarViscosityStatic();
	
	p_domainCPU->C.value = C->value.data_ptr<scalar_t>();
	p_domainCPU->C.index = C->index.data_ptr<index_t>();
	p_domainCPU->C.row = C->row.data_ptr<index_t>();
	p_domainCPU->Adiag = A.data_ptr<scalar_t>();
	p_domainCPU->P.value = P->value.data_ptr<scalar_t>();
	p_domainCPU->P.index = P->index.data_ptr<index_t>();
	p_domainCPU->P.row = P->row.data_ptr<index_t>();
#ifdef WITH_GRAD
	p_domainCPU->viscosity_grad = getViscosityGradDataPtr<scalar_t>();
	p_domainCPU->scalarViscosity_grad = hasPassiveScalarViscosity() ? getPassiveScalarViscosityGradDataPtr<scalar_t>() : p_domainCPU->viscosity_grad;

	p_domainCPU->C_grad.value = C_grad->value.data_ptr<scalar_t>();
	p_domainCPU->C_grad.index = C_grad->index.data_ptr<index_t>();
	p_domainCPU->C_grad.row = C_grad->row.data_ptr<index_t>();
	p_domainCPU->Adiag_grad = getTensorDataPtr<scalar_t>(A_grad);
	p_domainCPU->P_grad.value = P_grad->value.data_ptr<scalar_t>();
	p_domainCPU->P_grad.index = P_grad->index.data_ptr<index_t>();
	p_domainCPU->P_grad.row = P_grad->row.data_ptr<index_t>();
#endif
	
	if(hasPassiveScalar()){
		p_domainCPU->scalarRHS = scalarRHS.data_ptr<scalar_t>();
		p_domainCPU->scalarResult = scalarResult.data_ptr<scalar_t>();
	} else {
		p_domainCPU->scalarRHS = nullptr;
		p_domainCPU->scalarResult = nullptr;
	}
	p_domainCPU->velocityRHS = velocityRHS.data_ptr<scalar_t>();
	p_domainCPU->velocityResult = velocityResult.data_ptr<scalar_t>();
	p_domainCPU->pressureRHS = pressureRHS.data_ptr<scalar_t>();
	p_domainCPU->pressureRHSdiv = pressureRHSdiv.data_ptr<scalar_t>();
	p_domainCPU->pressureResult = pressureResult.data_ptr<scalar_t>();
	
#ifdef WITH_GRAD
	if(hasPassiveScalar()){
		p_domainCPU->scalarRHS_grad = getTensorDataPtr<scalar_t>(scalarRHS_grad);
		p_domainCPU->scalarResult_grad = getTensorDataPtr<scalar_t>(scalarResult_grad);
	} else {
		p_domainCPU->scalarRHS_grad = nullptr;
		p_domainCPU->scalarResult_grad = nullptr;
	}
	p_domainCPU->velocityRHS_grad = getTensorDataPtr<scalar_t>(velocityRHS_grad);
	p_domainCPU->velocityResult_grad = getTensorDataPtr<scalar_t>(velocityResult_grad);
	p_domainCPU->pressureRHS_grad = getTensorDataPtr<scalar_t>(pressureRHS_grad);
	p_domainCPU->pressureRHSdiv_grad = getTensorDataPtr<scalar_t>(pressureRHSdiv_grad);
	p_domainCPU->pressureResult_grad = getTensorDataPtr<scalar_t>(pressureResult_grad);
#endif
	
	for(index_t blockIdx=0, numBlocks=blocks.size(); blockIdx<numBlocks; ++blockIdx){
		BlockGPU<scalar_t> *p_blockCPU = p_blocksCPU + blockIdx;
		std::shared_ptr<const Block> block = blocks[blockIdx];
		
		TORCH_CHECK(block->hasPassiveScalar()==hasPassiveScalar(), "Inconsistent use of passive scalar.")
		TORCH_CHECK(block->getPassiveScalarChannels()==getPassiveScalarChannels(), "Inconsistent number of channels in passive scalar.")
		
		p_blockCPU->globalOffset = block->globalOffset; //offset in cells from first block start. used e.g. for CSR indices
		p_blockCPU->csrOffset = block->csrOffset;
		p_blockCPU->size = block->getSizes();
		p_blockCPU->stride = block->getStrides();
		p_blockCPU->viscosity = block->getViscosityDataPtr<scalar_t>();
		p_blockCPU->isViscosityStatic = block->isViscosityStatic();
		p_blockCPU->velocity = block->velocity.data_ptr<scalar_t>();
		p_blockCPU->velocitySource = block->getVelocitySourceDataPtr<scalar_t>();
		p_blockCPU->isVelocitySourceStatic = block->velocitySourceStatic;
		p_blockCPU->pressure = block->pressure.data_ptr<scalar_t>();
		p_blockCPU->scalarData = block->getPassiveScalarDataPtr<scalar_t>(); //passiveScalar.data_ptr<scalar_t>();
#ifdef WITH_GRAD
		p_blockCPU->viscosity_grad = block->getViscosityGradDataPtr<scalar_t>();
		p_blockCPU->velocity_grad = getTensorDataPtr<scalar_t>(block->velocity_grad);
		p_blockCPU->velocitySource_grad = block->getVelocitySourceGradDataPtr<scalar_t>();
		p_blockCPU->pressure_grad = getTensorDataPtr<scalar_t>(block->pressure_grad);
		p_blockCPU->scalarData_grad = getTensorDataPtr<scalar_t>(block->passiveScalar_grad);
#endif
		//scalar_t *velocityUpdate;
		for(index_t boundIdx=0; boundIdx<getSpatialDims()*2; ++boundIdx){
			const std::shared_ptr<Boundary> boundary = block->boundaries[boundIdx];
			p_blockCPU->boundaries[boundIdx].type = boundary->type;
			switch(boundary->type){
			case BoundaryType::FIXED:
			{
				std::shared_ptr<const FixedBoundary> bound = std::static_pointer_cast<const FixedBoundary>(boundary);
				
				TORCH_CHECK(block->hasPassiveScalar()==bound->hasPassiveScalar(), "Inconsistent use of passive scalar.")
				TORCH_CHECK(block->getPassiveScalarChannels()==bound->getPassiveScalarChannels(), "Inconsistent number of channels in passive scalar.")
				
				FixedBoundaryGPU<scalar_t> fb;
				memset(&fb, 0, sizeof(FixedBoundaryGPU<scalar_t>));
				fb.size = bound->getSizes();
				fb.stride = bound->getStrides();
				
				if(bound->isPassiveScalarBoundaryTypeStatic()){
					fb.passiveScalar.boundaryType = bound->m_passiveScalarTypes ? bound->m_passiveScalarTypes.value()[0] : BoundaryConditionType::DIRICHLET;
					fb.passiveScalar.isStaticType = true;
				}else{
					fb.passiveScalar.p_boundaryTypes = reinterpret_cast<BoundaryConditionType*>(bound->getPassiveScalarTypesDataPtr<BoundaryConditionType_base_type>());
					//fb.passiveScalar.p_boundaryTypes = bound->getPassiveScalarTypesDataPtr<BoundaryConditionType>();
					fb.passiveScalar.isStaticType = false;
				}
				//fb.passiveScalar.boundaryType = bound->m_passiveScalarType;
				fb.passiveScalar.data = bound->getPassiveScalarDataPtr<scalar_t>();
				fb.passiveScalar.isStatic = bound->m_passiveScalarStatic;
				
				fb.velocity.boundaryType = bound->m_velocityType;
				fb.velocity.isStaticType = true;
				fb.velocity.data = bound->getVelocityDataPtr<scalar_t>();
				fb.velocity.isStatic = bound->m_velocityStatic;
				
				fb.pressure.boundaryType = bound->m_pressureType;
				fb.pressure.isStaticType = true;
				fb.pressure.data = bound->getPressureDataPtr<scalar_t>();
				fb.pressure.isStatic = bound->m_pressureStatic;
				
#ifdef WITH_GRAD
				fb.passiveScalar.grad = bound->getPassiveScalarGradDataPtr<scalar_t>();
				fb.velocity.grad = getTensorDataPtr<scalar_t>(bound->m_velocity_grad);
				fb.pressure.grad = nullptr;
#endif
				
				fb.hasTransform = bound->hasTransform();
				fb.transform = bound->getTransformDataPtr<scalar_t>();
				p_blockCPU->boundaries[boundIdx].fb = fb;
				//TORCH_CHECK(false, "FixedBoundary is not yet supported by the simulator.");
				break;
			}
			case BoundaryType::DIRICHLET:
			{
				std::shared_ptr<const StaticDirichletBoundary> bound = std::static_pointer_cast<const StaticDirichletBoundary>(boundary);
				StaticDirichletBoundaryGPU<scalar_t> sdb;
				memset(&sdb, 0, sizeof(StaticDirichletBoundaryGPU<scalar_t>));
				sdb.slip = bound->slip.data_ptr<scalar_t>()[0];
				memcpy(&sdb.velocity.a, bound->boundaryVelocity.data_ptr<scalar_t>(), sizeof(scalar_t)*getSpatialDims());
				sdb.scalar = bound->boundaryScalar.data_ptr<scalar_t>()[0];
#ifdef WITH_GRAD
				//vdb.slip_grad = getTensorDataPtr<scalar_t>(bound->slip_grad);
				sdb.velocity_grad = getTensorDataPtr<scalar_t>(bound->boundaryVelocity_grad);
				sdb.scalar_grad = getTensorDataPtr<scalar_t>(bound->boundaryScalar_grad);
#endif
				//sdb.hasTransform = bound->hasTransform;
				//sdb.transform = bound->hasTransform ? bound->transform.data_ptr<scalar_t>() : nullptr;
				p_blockCPU->boundaries[boundIdx].sdb = sdb;
				break;
			}
			case BoundaryType::DIRICHLET_VARYING:
			{
				std::shared_ptr<const VaryingDirichletBoundary> bound = std::static_pointer_cast<const VaryingDirichletBoundary>(boundary);
				VaryingDirichletBoundaryGPU<scalar_t> vdb;
				memset(&vdb, 0, sizeof(VaryingDirichletBoundaryGPU<scalar_t>));
				vdb.slip = bound->slip.data_ptr<scalar_t>()[0];
				vdb.velocity = bound->boundaryVelocity.data_ptr<scalar_t>();
				vdb.scalar = bound->boundaryScalar.data_ptr<scalar_t>();
				vdb.size = bound->getSizes();
				vdb.stride = bound->getStrides();
#ifdef WITH_GRAD
				//vdb.slip_grad = getTensorDataPtr<scalar_t>(bound->slip_grad);
				vdb.velocity_grad = getTensorDataPtr<scalar_t>(bound->boundaryVelocity_grad);
				vdb.scalar_grad = getTensorDataPtr<scalar_t>(bound->boundaryScalar_grad);
#endif
				vdb.hasTransform = bound->hasTransform;
				vdb.transform = bound->hasTransform ? bound->transform.data_ptr<scalar_t>() : nullptr;
				p_blockCPU->boundaries[boundIdx].vdb = vdb;
				break;
			}
			case BoundaryType::NEUMANN:
			{
				//std::shared_ptr<const StaticNeumannBoundary> bound = std::static_pointer_cast<const StaticNeumannBoundary>(boundary);
				StaticNeumannBoundaryGPU<scalar_t> snb;
				memset(&snb, 0, sizeof(StaticNeumannBoundaryGPU<scalar_t>));
				p_blockCPU->boundaries[boundIdx].snb = snb;
				break;
			}
			case BoundaryType::CONNECTED_GRID:
			{
				std::shared_ptr<const ConnectedBoundary> bound = std::static_pointer_cast<const ConnectedBoundary>(boundary);
				ConnectedBoundaryGPU<scalar_t> cb;
				memset(&cb, 0, sizeof(ConnectedBoundaryGPU<scalar_t>));
				cb.connectedGridIndex = getBlockIdx(bound->getConnectedBlock());
				const index_t numAxes = static_cast<index_t>(bound->axes.size());
				for(index_t axisIdx=0; axisIdx<numAxes; ++axisIdx){
					cb.axes.a[axisIdx] = bound->axes[axisIdx];
				}
				p_blockCPU->boundaries[boundIdx].cb = cb;
				break;
			}
			case BoundaryType::PERIODIC:
			{
				std::shared_ptr<const PeriodicBoundary> bound = std::static_pointer_cast<const PeriodicBoundary>(boundary);
				PeriodicBoundaryGPU<scalar_t> pb;
				memset(&pb, 0, sizeof(PeriodicBoundaryGPU<scalar_t>));
				
				p_blockCPU->boundaries[boundIdx].pb = pb;
				break;
			}
			default:
				TORCH_CHECK(false, "Unknown boundary encountered in Domain::UpdateDomainGPU.");
				break;
			}
		}
		p_blockCPU->hasTransform = block->hasTransform();
		p_blockCPU->transform = block->getTransformDataPtr<scalar_t>(); //block->hasTransform() ? block->transform.data_ptr<scalar_t>() : nullptr;
		p_blockCPU->hasFaceTransform = block->hasFaceTransform();
		p_blockCPU->faceTransform = block->getFaceTransformDataPtr<scalar_t>(); //block->hasFaceTransform() ? block->faceTransform.data_ptr<scalar_t>() : nullptr;
	}
	
	CopyDomainToGPU(atlas);
	//set the pointers for host memory
	reinterpret_cast<DomainGPU<scalar_t>*>(atlas.p_host)->blocks = reinterpret_cast<BlockGPU<scalar_t>*>(reinterpret_cast<uint8_t*>(atlas.p_host) + atlas.blocksOffsetBytes);
	
	setTensorChanged(false);
}

index_t Domain::getBlockIdx(const std::shared_ptr<const Block> block) const {
	auto it = find(blocks.begin(), blocks.end(), block);
	if(it==blocks.end()) return -1;
	return it - blocks.begin();
}

torch::Dtype Domain::getDtype() const {
	return m_dtype;
    //TORCH_CHECK(getNumBlocks()>0, "Domain does not contain any blocks.")
    //return blocks[0]->getDtype();
}

py::object Domain::getPyDtype() const {
	//return (PyObject*)torch::getTHPDtype(getDtype());
	if(pyDtype.has_value()){
		return pyDtype.value();
	} else {
		TORCH_CHECK(false, "No python dtype set.");
	}
}

torch::Device Domain::getDevice() const {
	return m_device;
    //TORCH_CHECK(getNumBlocks()>0, "Domain does not contain any blocks.")
    //return blocks[0]->getDevice();
}

torch::TensorOptions Domain::getTensorOptions() const {
	return valueOptions;
}

index_t Domain::getSpatialDims() const {
	return m_spatialDims;
    //TORCH_CHECK(getNumBlocks()>0, "Domain does not contain any blocks.")
    //return blocks[0]->getSpatialDims();
}
index_t Domain::getTotalSize() const {
	TORCH_CHECK(initialized, "domain is not initialized.");
	return totalSize;
}

index_t Domain::getMaxBlockSize() const {
    TORCH_CHECK(getNumBlocks()>0, "Domain does not contain any blocks.")
	index_t maxSize = 0;
	for(auto p_block : blocks){
		const index_t blockSize = p_block->getStrides().w;
		if(blockSize>maxSize){
			maxSize = blockSize;
		}
	}
	return maxSize;
}

bool Domain::IsTensorChanged() const {
	if(isTensorChanged){ return true; }
	if(C->IsTensorChanged()){ return true; }
	if(P->IsTensorChanged()){ return true; }
#ifdef WITH_GRAD
	if(C_grad->IsTensorChanged()){ return true; }
	if(P_grad->IsTensorChanged()){ return true; }
#endif
	for(auto block : blocks){
		if(block->IsTensorChanged()){ return true; }
	}
	return false;
}
void Domain::setTensorChanged(const bool changed){
	isTensorChanged = changed;
	C->setTensorChanged(changed);
	P->setTensorChanged(changed);
#ifdef WITH_GRAD
	C_grad->setTensorChanged(changed);
	P_grad->setTensorChanged(changed);
#endif
	for(auto block : blocks){
		block->setTensorChanged(changed);
	}
	
}

std::string Domain::ToString() const {
	std::ostringstream repr;
	repr << "Domain(\"" << name << "\" ";
	//if(blocks.size()>0) repr << getSpatialDims();
	//else repr << "?";
	repr << getSpatialDims() << "D";
	repr << ", scalarChannels=" << getPassiveScalarChannels();
	repr << ", blocks=[";
	//for(auto block : blocks){
	for(auto blockIt = blocks.begin(); blockIt!=blocks.end(); ++blockIt){
		repr << (*blockIt)->name;
		if(blockIt!=(blocks.end()-1)) repr << ", ";
	}
	repr << "], initialized=" << initialized << " )";
	return repr.str();
	
}
