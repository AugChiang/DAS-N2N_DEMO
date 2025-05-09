�
��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
�
BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
�
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

$
DisableCopyOnRead
resource�
.
Identity

input"T
output"T"	
Ttype
\
	LeakyRelu
features"T
activations"T"
alphafloat%��L>"
Ttype0:
2
�
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �
?
Mul
x"T
y"T
z"T"
Ttype:
2	�

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
�
ResizeNearestNeighbor
images"T
size
resized_images"T"
Ttype:
2
	"
align_cornersbool( "
half_pixel_centersbool( 
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
d
Shape

input"T&
output"out_type��out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.12.02v2.12.0-rc1-12-g0db597d0d758��
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
f
	iterationVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	
l

out01/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
out01/bias
e
out01/bias/Read/ReadVariableOpReadVariableOp
out01/bias*
_output_shapes
:*
dtype0
|
out01/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*
shared_nameout01/kernel
u
 out01/kernel/Read/ReadVariableOpReadVariableOpout01/kernel*&
_output_shapes
:0*
dtype0
p
conv01b/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*
shared_nameconv01b/bias
i
 conv01b/bias/Read/ReadVariableOpReadVariableOpconv01b/bias*
_output_shapes
:0*
dtype0
�
conv01b/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:00*
shared_nameconv01b/kernel
y
"conv01b/kernel/Read/ReadVariableOpReadVariableOpconv01b/kernel*&
_output_shapes
:00*
dtype0
p
conv01a/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*
shared_nameconv01a/bias
i
 conv01a/bias/Read/ReadVariableOpReadVariableOpconv01a/bias*
_output_shapes
:0*
dtype0
�
conv01a/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:00*
shared_nameconv01a/kernel
y
"conv01a/kernel/Read/ReadVariableOpReadVariableOpconv01a/kernel*&
_output_shapes
:00*
dtype0
n
conv10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv10/bias
g
conv10/bias/Read/ReadVariableOpReadVariableOpconv10/bias*
_output_shapes
:*
dtype0
~
conv10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv10/kernel
w
!conv10/kernel/Read/ReadVariableOpReadVariableOpconv10/kernel*&
_output_shapes
:*
dtype0
n
conv00/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv00/bias
g
conv00/bias/Read/ReadVariableOpReadVariableOpconv00/bias*
_output_shapes
:*
dtype0
~
conv00/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv00/kernel
w
!conv00/kernel/Read/ReadVariableOpReadVariableOpconv00/kernel*&
_output_shapes
:*
dtype0
�
serving_default_input_layerPlaceholder*,
_output_shapes
:����������`*
dtype0*!
shape:����������`
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_layerconv00/kernelconv00/biasconv10/kernelconv10/biasconv01a/kernelconv01a/biasconv01b/kernelconv01b/biasout01/kernel
out01/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������`*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8� *+
f&R$
"__inference_signature_wrapper_6774

NoOpNoOp
�Q
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�P
value�PB�P B�P
�
layer-0
layer-1
layer_with_weights-0
layer-2
layer-3
layer-4
layer_with_weights-1
layer-5
layer-6
layer-7
	layer-8

layer_with_weights-2

layer-9
layer-10
layer_with_weights-3
layer-11
layer-12
layer_with_weights-4
layer-13
layer-14
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
* 
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses* 
�
	variables
 trainable_variables
!regularization_losses
"	keras_api
#__call__
*$&call_and_return_all_conditional_losses

%kernel
&bias
 '_jit_compiled_convolution_op*
�
(	variables
)trainable_variables
*regularization_losses
+	keras_api
,__call__
*-&call_and_return_all_conditional_losses* 
�
.	variables
/trainable_variables
0regularization_losses
1	keras_api
2__call__
*3&call_and_return_all_conditional_losses* 
�
4	variables
5trainable_variables
6regularization_losses
7	keras_api
8__call__
*9&call_and_return_all_conditional_losses

:kernel
;bias
 <_jit_compiled_convolution_op*
�
=	variables
>trainable_variables
?regularization_losses
@	keras_api
A__call__
*B&call_and_return_all_conditional_losses* 
�
C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
G__call__
*H&call_and_return_all_conditional_losses* 
�
I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
M__call__
*N&call_and_return_all_conditional_losses* 
�
O	variables
Ptrainable_variables
Qregularization_losses
R	keras_api
S__call__
*T&call_and_return_all_conditional_losses

Ukernel
Vbias
 W_jit_compiled_convolution_op*
�
X	variables
Ytrainable_variables
Zregularization_losses
[	keras_api
\__call__
*]&call_and_return_all_conditional_losses* 
�
^	variables
_trainable_variables
`regularization_losses
a	keras_api
b__call__
*c&call_and_return_all_conditional_losses

dkernel
ebias
 f_jit_compiled_convolution_op*
�
g	variables
htrainable_variables
iregularization_losses
j	keras_api
k__call__
*l&call_and_return_all_conditional_losses* 
�
m	variables
ntrainable_variables
oregularization_losses
p	keras_api
q__call__
*r&call_and_return_all_conditional_losses

skernel
tbias
 u_jit_compiled_convolution_op*
�
v	variables
wtrainable_variables
xregularization_losses
y	keras_api
z__call__
*{&call_and_return_all_conditional_losses* 
J
%0
&1
:2
;3
U4
V5
d6
e7
s8
t9*
J
%0
&1
:2
;3
U4
V5
d6
e7
s8
t9*
* 
�
|non_trainable_variables

}layers
~metrics
layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
:
�trace_0
�trace_1
�trace_2
�trace_3* 
:
�trace_0
�trace_1
�trace_2
�trace_3* 
* 
S
�
_variables
�_iterations
�_learning_rate
�_update_step_xla*

�serving_default* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

%0
&1*

%0
&1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
 trainable_variables
!regularization_losses
#__call__
*$&call_and_return_all_conditional_losses
&$"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
]W
VARIABLE_VALUEconv00/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEconv00/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
(	variables
)trainable_variables
*regularization_losses
,__call__
*-&call_and_return_all_conditional_losses
&-"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
.	variables
/trainable_variables
0regularization_losses
2__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

:0
;1*

:0
;1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
4	variables
5trainable_variables
6regularization_losses
8__call__
*9&call_and_return_all_conditional_losses
&9"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
]W
VARIABLE_VALUEconv10/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEconv10/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
=	variables
>trainable_variables
?regularization_losses
A__call__
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
C	variables
Dtrainable_variables
Eregularization_losses
G__call__
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
I	variables
Jtrainable_variables
Kregularization_losses
M__call__
*N&call_and_return_all_conditional_losses
&N"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

U0
V1*

U0
V1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
O	variables
Ptrainable_variables
Qregularization_losses
S__call__
*T&call_and_return_all_conditional_losses
&T"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
^X
VARIABLE_VALUEconv01a/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEconv01a/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
X	variables
Ytrainable_variables
Zregularization_losses
\__call__
*]&call_and_return_all_conditional_losses
&]"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

d0
e1*

d0
e1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
^	variables
_trainable_variables
`regularization_losses
b__call__
*c&call_and_return_all_conditional_losses
&c"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
^X
VARIABLE_VALUEconv01b/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEconv01b/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
g	variables
htrainable_variables
iregularization_losses
k__call__
*l&call_and_return_all_conditional_losses
&l"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

s0
t1*

s0
t1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
m	variables
ntrainable_variables
oregularization_losses
q__call__
*r&call_and_return_all_conditional_losses
&r"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
\V
VARIABLE_VALUEout01/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
out01/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
v	variables
wtrainable_variables
xregularization_losses
z__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 
r
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14*

�0*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

�0*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
�	variables
�	keras_api

�total

�count*

�0
�1*

�	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameconv00/kernelconv00/biasconv10/kernelconv10/biasconv01a/kernelconv01a/biasconv01b/kernelconv01b/biasout01/kernel
out01/bias	iterationlearning_ratetotalcountConst*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *&
f!R
__inference__traced_save_7271
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv00/kernelconv00/biasconv10/kernelconv10/biasconv01a/kernelconv01a/biasconv01b/kernelconv01b/biasout01/kernel
out01/bias	iterationlearning_ratetotalcount*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *)
f$R"
 __inference__traced_restore_7323��
�P
�
__inference__wrapped_model_6251
input_layerE
+model_conv00_conv2d_readvariableop_resource::
,model_conv00_biasadd_readvariableop_resource:E
+model_conv10_conv2d_readvariableop_resource::
,model_conv10_biasadd_readvariableop_resource:F
,model_conv01a_conv2d_readvariableop_resource:00;
-model_conv01a_biasadd_readvariableop_resource:0F
,model_conv01b_conv2d_readvariableop_resource:00;
-model_conv01b_biasadd_readvariableop_resource:0D
*model_out01_conv2d_readvariableop_resource:09
+model_out01_biasadd_readvariableop_resource:
identity��#model/conv00/BiasAdd/ReadVariableOp�"model/conv00/Conv2D/ReadVariableOp�$model/conv01a/BiasAdd/ReadVariableOp�#model/conv01a/Conv2D/ReadVariableOp�$model/conv01b/BiasAdd/ReadVariableOp�#model/conv01b/Conv2D/ReadVariableOp�#model/conv10/BiasAdd/ReadVariableOp�"model/conv10/Conv2D/ReadVariableOp�"model/out01/BiasAdd/ReadVariableOp�!model/out01/Conv2D/ReadVariableOp^
model/reshape_1/ShapeShapeinput_layer*
T0*
_output_shapes
::��m
#model/reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%model/reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%model/reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
model/reshape_1/strided_sliceStridedSlicemodel/reshape_1/Shape:output:0,model/reshape_1/strided_slice/stack:output:0.model/reshape_1/strided_slice/stack_1:output:0.model/reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskb
model/reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :�a
model/reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :`a
model/reshape_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
model/reshape_1/Reshape/shapePack&model/reshape_1/strided_slice:output:0(model/reshape_1/Reshape/shape/1:output:0(model/reshape_1/Reshape/shape/2:output:0(model/reshape_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
model/reshape_1/ReshapeReshapeinput_layer&model/reshape_1/Reshape/shape:output:0*
T0*0
_output_shapes
:����������`�
"model/conv00/Conv2D/ReadVariableOpReadVariableOp+model_conv00_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
model/conv00/Conv2DConv2D model/reshape_1/Reshape:output:0*model/conv00/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������`*
paddingSAME*
strides
�
#model/conv00/BiasAdd/ReadVariableOpReadVariableOp,model_conv00_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model/conv00/BiasAddBiasAddmodel/conv00/Conv2D:output:0+model/conv00/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������`�
model/act00/LeakyRelu	LeakyRelumodel/conv00/BiasAdd:output:0*0
_output_shapes
:����������`*
alpha%���=�
model/down10/MaxPoolMaxPool#model/act00/LeakyRelu:activations:0*/
_output_shapes
:���������@0*
ksize
*
paddingSAME*
strides
�
"model/conv10/Conv2D/ReadVariableOpReadVariableOp+model_conv10_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
model/conv10/Conv2DConv2Dmodel/down10/MaxPool:output:0*model/conv10/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@0*
paddingSAME*
strides
�
#model/conv10/BiasAdd/ReadVariableOpReadVariableOp,model_conv10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model/conv10/BiasAddBiasAddmodel/conv10/Conv2D:output:0+model/conv10/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@0�
model/act10/LeakyRelu	LeakyRelumodel/conv10/BiasAdd:output:0*/
_output_shapes
:���������@0*
alpha%���=a
model/up01/ConstConst*
_output_shapes
:*
dtype0*
valueB"@   0   c
model/up01/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      r
model/up01/mulMulmodel/up01/Const:output:0model/up01/Const_1:output:0*
T0*
_output_shapes
:�
'model/up01/resize/ResizeNearestNeighborResizeNearestNeighbor#model/act10/LeakyRelu:activations:0model/up01/mul:z:0*
T0*0
_output_shapes
:����������`*
half_pixel_centers(\
model/concat01/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
model/concat01/concatConcatV2#model/act00/LeakyRelu:activations:08model/up01/resize/ResizeNearestNeighbor:resized_images:0#model/concat01/concat/axis:output:0*
N*
T0*0
_output_shapes
:����������`0�
#model/conv01a/Conv2D/ReadVariableOpReadVariableOp,model_conv01a_conv2d_readvariableop_resource*&
_output_shapes
:00*
dtype0�
model/conv01a/Conv2DConv2Dmodel/concat01/concat:output:0+model/conv01a/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������`0*
paddingSAME*
strides
�
$model/conv01a/BiasAdd/ReadVariableOpReadVariableOp-model_conv01a_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0�
model/conv01a/BiasAddBiasAddmodel/conv01a/Conv2D:output:0,model/conv01a/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������`0�
model/act01a/LeakyRelu	LeakyRelumodel/conv01a/BiasAdd:output:0*0
_output_shapes
:����������`0*
alpha%���=�
#model/conv01b/Conv2D/ReadVariableOpReadVariableOp,model_conv01b_conv2d_readvariableop_resource*&
_output_shapes
:00*
dtype0�
model/conv01b/Conv2DConv2D$model/act01a/LeakyRelu:activations:0+model/conv01b/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������`0*
paddingSAME*
strides
�
$model/conv01b/BiasAdd/ReadVariableOpReadVariableOp-model_conv01b_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0�
model/conv01b/BiasAddBiasAddmodel/conv01b/Conv2D:output:0,model/conv01b/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������`0�
model/act01b/LeakyRelu	LeakyRelumodel/conv01b/BiasAdd:output:0*0
_output_shapes
:����������`0*
alpha%���=�
!model/out01/Conv2D/ReadVariableOpReadVariableOp*model_out01_conv2d_readvariableop_resource*&
_output_shapes
:0*
dtype0�
model/out01/Conv2DConv2D$model/act01b/LeakyRelu:activations:0)model/out01/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������`*
paddingSAME*
strides
�
"model/out01/BiasAdd/ReadVariableOpReadVariableOp+model_out01_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model/out01/BiasAddBiasAddmodel/out01/Conv2D:output:0*model/out01/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������`n
model/out01_1d/ShapeShapemodel/out01/BiasAdd:output:0*
T0*
_output_shapes
::��l
"model/out01_1d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$model/out01_1d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$model/out01_1d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
model/out01_1d/strided_sliceStridedSlicemodel/out01_1d/Shape:output:0+model/out01_1d/strided_slice/stack:output:0-model/out01_1d/strided_slice/stack_1:output:0-model/out01_1d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maska
model/out01_1d/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :�`
model/out01_1d/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :`�
model/out01_1d/Reshape/shapePack%model/out01_1d/strided_slice:output:0'model/out01_1d/Reshape/shape/1:output:0'model/out01_1d/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:�
model/out01_1d/ReshapeReshapemodel/out01/BiasAdd:output:0%model/out01_1d/Reshape/shape:output:0*
T0*,
_output_shapes
:����������`s
IdentityIdentitymodel/out01_1d/Reshape:output:0^NoOp*
T0*,
_output_shapes
:����������`�
NoOpNoOp$^model/conv00/BiasAdd/ReadVariableOp#^model/conv00/Conv2D/ReadVariableOp%^model/conv01a/BiasAdd/ReadVariableOp$^model/conv01a/Conv2D/ReadVariableOp%^model/conv01b/BiasAdd/ReadVariableOp$^model/conv01b/Conv2D/ReadVariableOp$^model/conv10/BiasAdd/ReadVariableOp#^model/conv10/Conv2D/ReadVariableOp#^model/out01/BiasAdd/ReadVariableOp"^model/out01/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:����������`: : : : : : : : : : 2J
#model/conv00/BiasAdd/ReadVariableOp#model/conv00/BiasAdd/ReadVariableOp2H
"model/conv00/Conv2D/ReadVariableOp"model/conv00/Conv2D/ReadVariableOp2L
$model/conv01a/BiasAdd/ReadVariableOp$model/conv01a/BiasAdd/ReadVariableOp2J
#model/conv01a/Conv2D/ReadVariableOp#model/conv01a/Conv2D/ReadVariableOp2L
$model/conv01b/BiasAdd/ReadVariableOp$model/conv01b/BiasAdd/ReadVariableOp2J
#model/conv01b/Conv2D/ReadVariableOp#model/conv01b/Conv2D/ReadVariableOp2J
#model/conv10/BiasAdd/ReadVariableOp#model/conv10/BiasAdd/ReadVariableOp2H
"model/conv10/Conv2D/ReadVariableOp"model/conv10/Conv2D/ReadVariableOp2H
"model/out01/BiasAdd/ReadVariableOp"model/out01/BiasAdd/ReadVariableOp2F
!model/out01/Conv2D/ReadVariableOp!model/out01/Conv2D/ReadVariableOp:Y U
,
_output_shapes
:����������`
%
_user_specified_nameinput_layer
�
C
'__inference_out01_1d_layer_call_fn_7151

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_out01_1d_layer_call_and_return_conditional_losses_6434e
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:����������`"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������`:X T
0
_output_shapes
:����������`
 
_user_specified_nameinputs
�I
�
?__inference_model_layer_call_and_return_conditional_losses_6888

inputs?
%conv00_conv2d_readvariableop_resource:4
&conv00_biasadd_readvariableop_resource:?
%conv10_conv2d_readvariableop_resource:4
&conv10_biasadd_readvariableop_resource:@
&conv01a_conv2d_readvariableop_resource:005
'conv01a_biasadd_readvariableop_resource:0@
&conv01b_conv2d_readvariableop_resource:005
'conv01b_biasadd_readvariableop_resource:0>
$out01_conv2d_readvariableop_resource:03
%out01_biasadd_readvariableop_resource:
identity��conv00/BiasAdd/ReadVariableOp�conv00/Conv2D/ReadVariableOp�conv01a/BiasAdd/ReadVariableOp�conv01a/Conv2D/ReadVariableOp�conv01b/BiasAdd/ReadVariableOp�conv01b/Conv2D/ReadVariableOp�conv10/BiasAdd/ReadVariableOp�conv10/Conv2D/ReadVariableOp�out01/BiasAdd/ReadVariableOp�out01/Conv2D/ReadVariableOpS
reshape_1/ShapeShapeinputs*
T0*
_output_shapes
::��g
reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: i
reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
reshape_1/strided_sliceStridedSlicereshape_1/Shape:output:0&reshape_1/strided_slice/stack:output:0(reshape_1/strided_slice/stack_1:output:0(reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :�[
reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :`[
reshape_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
reshape_1/Reshape/shapePack reshape_1/strided_slice:output:0"reshape_1/Reshape/shape/1:output:0"reshape_1/Reshape/shape/2:output:0"reshape_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
reshape_1/ReshapeReshapeinputs reshape_1/Reshape/shape:output:0*
T0*0
_output_shapes
:����������`�
conv00/Conv2D/ReadVariableOpReadVariableOp%conv00_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
conv00/Conv2DConv2Dreshape_1/Reshape:output:0$conv00/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������`*
paddingSAME*
strides
�
conv00/BiasAdd/ReadVariableOpReadVariableOp&conv00_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv00/BiasAddBiasAddconv00/Conv2D:output:0%conv00/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������`w
act00/LeakyRelu	LeakyReluconv00/BiasAdd:output:0*0
_output_shapes
:����������`*
alpha%���=�
down10/MaxPoolMaxPoolact00/LeakyRelu:activations:0*/
_output_shapes
:���������@0*
ksize
*
paddingSAME*
strides
�
conv10/Conv2D/ReadVariableOpReadVariableOp%conv10_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
conv10/Conv2DConv2Ddown10/MaxPool:output:0$conv10/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@0*
paddingSAME*
strides
�
conv10/BiasAdd/ReadVariableOpReadVariableOp&conv10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv10/BiasAddBiasAddconv10/Conv2D:output:0%conv10/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@0v
act10/LeakyRelu	LeakyReluconv10/BiasAdd:output:0*/
_output_shapes
:���������@0*
alpha%���=[

up01/ConstConst*
_output_shapes
:*
dtype0*
valueB"@   0   ]
up01/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      `
up01/mulMulup01/Const:output:0up01/Const_1:output:0*
T0*
_output_shapes
:�
!up01/resize/ResizeNearestNeighborResizeNearestNeighboract10/LeakyRelu:activations:0up01/mul:z:0*
T0*0
_output_shapes
:����������`*
half_pixel_centers(V
concat01/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concat01/concatConcatV2act00/LeakyRelu:activations:02up01/resize/ResizeNearestNeighbor:resized_images:0concat01/concat/axis:output:0*
N*
T0*0
_output_shapes
:����������`0�
conv01a/Conv2D/ReadVariableOpReadVariableOp&conv01a_conv2d_readvariableop_resource*&
_output_shapes
:00*
dtype0�
conv01a/Conv2DConv2Dconcat01/concat:output:0%conv01a/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������`0*
paddingSAME*
strides
�
conv01a/BiasAdd/ReadVariableOpReadVariableOp'conv01a_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0�
conv01a/BiasAddBiasAddconv01a/Conv2D:output:0&conv01a/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������`0y
act01a/LeakyRelu	LeakyReluconv01a/BiasAdd:output:0*0
_output_shapes
:����������`0*
alpha%���=�
conv01b/Conv2D/ReadVariableOpReadVariableOp&conv01b_conv2d_readvariableop_resource*&
_output_shapes
:00*
dtype0�
conv01b/Conv2DConv2Dact01a/LeakyRelu:activations:0%conv01b/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������`0*
paddingSAME*
strides
�
conv01b/BiasAdd/ReadVariableOpReadVariableOp'conv01b_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0�
conv01b/BiasAddBiasAddconv01b/Conv2D:output:0&conv01b/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������`0y
act01b/LeakyRelu	LeakyReluconv01b/BiasAdd:output:0*0
_output_shapes
:����������`0*
alpha%���=�
out01/Conv2D/ReadVariableOpReadVariableOp$out01_conv2d_readvariableop_resource*&
_output_shapes
:0*
dtype0�
out01/Conv2DConv2Dact01b/LeakyRelu:activations:0#out01/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������`*
paddingSAME*
strides
~
out01/BiasAdd/ReadVariableOpReadVariableOp%out01_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
out01/BiasAddBiasAddout01/Conv2D:output:0$out01/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������`b
out01_1d/ShapeShapeout01/BiasAdd:output:0*
T0*
_output_shapes
::��f
out01_1d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: h
out01_1d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
out01_1d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
out01_1d/strided_sliceStridedSliceout01_1d/Shape:output:0%out01_1d/strided_slice/stack:output:0'out01_1d/strided_slice/stack_1:output:0'out01_1d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask[
out01_1d/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :�Z
out01_1d/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :`�
out01_1d/Reshape/shapePackout01_1d/strided_slice:output:0!out01_1d/Reshape/shape/1:output:0!out01_1d/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:�
out01_1d/ReshapeReshapeout01/BiasAdd:output:0out01_1d/Reshape/shape:output:0*
T0*,
_output_shapes
:����������`m
IdentityIdentityout01_1d/Reshape:output:0^NoOp*
T0*,
_output_shapes
:����������`�
NoOpNoOp^conv00/BiasAdd/ReadVariableOp^conv00/Conv2D/ReadVariableOp^conv01a/BiasAdd/ReadVariableOp^conv01a/Conv2D/ReadVariableOp^conv01b/BiasAdd/ReadVariableOp^conv01b/Conv2D/ReadVariableOp^conv10/BiasAdd/ReadVariableOp^conv10/Conv2D/ReadVariableOp^out01/BiasAdd/ReadVariableOp^out01/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:����������`: : : : : : : : : : 2>
conv00/BiasAdd/ReadVariableOpconv00/BiasAdd/ReadVariableOp2<
conv00/Conv2D/ReadVariableOpconv00/Conv2D/ReadVariableOp2@
conv01a/BiasAdd/ReadVariableOpconv01a/BiasAdd/ReadVariableOp2>
conv01a/Conv2D/ReadVariableOpconv01a/Conv2D/ReadVariableOp2@
conv01b/BiasAdd/ReadVariableOpconv01b/BiasAdd/ReadVariableOp2>
conv01b/Conv2D/ReadVariableOpconv01b/Conv2D/ReadVariableOp2>
conv10/BiasAdd/ReadVariableOpconv10/BiasAdd/ReadVariableOp2<
conv10/Conv2D/ReadVariableOpconv10/Conv2D/ReadVariableOp2<
out01/BiasAdd/ReadVariableOpout01/BiasAdd/ReadVariableOp2:
out01/Conv2D/ReadVariableOpout01/Conv2D/ReadVariableOp:T P
,
_output_shapes
:����������`
 
_user_specified_nameinputs
�
Z
>__inference_up01_layer_call_and_return_conditional_losses_7056

inputs
identityI
ShapeShapeinputs*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:V
ConstConst*
_output_shapes
:*
dtype0*
valueB"      W
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:�
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4������������������������������������*
half_pixel_centers(�
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
\
@__inference_act01a_layer_call_and_return_conditional_losses_6380

inputs
identity`
	LeakyRelu	LeakyReluinputs*0
_output_shapes
:����������`0*
alpha%���=h
IdentityIdentityLeakyRelu:activations:0*
T0*0
_output_shapes
:����������`0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������`0:X T
0
_output_shapes
:����������`0
 
_user_specified_nameinputs
�4
�
?__inference_model_layer_call_and_return_conditional_losses_6437
input_layer%
conv00_6313:
conv00_6315:%
conv10_6337:
conv10_6339:&
conv01a_6370:00
conv01a_6372:0&
conv01b_6393:00
conv01b_6395:0$

out01_6416:0

out01_6418:
identity��conv00/StatefulPartitionedCall�conv01a/StatefulPartitionedCall�conv01b/StatefulPartitionedCall�conv10/StatefulPartitionedCall�out01/StatefulPartitionedCall�
reshape_1/PartitionedCallPartitionedCallinput_layer*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_reshape_1_layer_call_and_return_conditional_losses_6300�
conv00/StatefulPartitionedCallStatefulPartitionedCall"reshape_1/PartitionedCall:output:0conv00_6313conv00_6315*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������`*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_conv00_layer_call_and_return_conditional_losses_6312�
act00/PartitionedCallPartitionedCall'conv00/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *H
fCRA
?__inference_act00_layer_call_and_return_conditional_losses_6323�
down10/PartitionedCallPartitionedCallact00/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@0* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_down10_layer_call_and_return_conditional_losses_6257�
conv10/StatefulPartitionedCallStatefulPartitionedCalldown10/PartitionedCall:output:0conv10_6337conv10_6339*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_conv10_layer_call_and_return_conditional_losses_6336�
act10/PartitionedCallPartitionedCall'conv10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@0* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *H
fCRA
?__inference_act10_layer_call_and_return_conditional_losses_6347�
up01/PartitionedCallPartitionedCallact10/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *G
fBR@
>__inference_up01_layer_call_and_return_conditional_losses_6276�
concat01/PartitionedCallPartitionedCallact00/PartitionedCall:output:0up01/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������`0* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_concat01_layer_call_and_return_conditional_losses_6357�
conv01a/StatefulPartitionedCallStatefulPartitionedCall!concat01/PartitionedCall:output:0conv01a_6370conv01a_6372*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������`0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_conv01a_layer_call_and_return_conditional_losses_6369�
act01a/PartitionedCallPartitionedCall(conv01a/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������`0* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_act01a_layer_call_and_return_conditional_losses_6380�
conv01b/StatefulPartitionedCallStatefulPartitionedCallact01a/PartitionedCall:output:0conv01b_6393conv01b_6395*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������`0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_conv01b_layer_call_and_return_conditional_losses_6392�
act01b/PartitionedCallPartitionedCall(conv01b/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������`0* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_act01b_layer_call_and_return_conditional_losses_6403�
out01/StatefulPartitionedCallStatefulPartitionedCallact01b/PartitionedCall:output:0
out01_6416
out01_6418*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������`*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *H
fCRA
?__inference_out01_layer_call_and_return_conditional_losses_6415�
out01_1d/PartitionedCallPartitionedCall&out01/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_out01_1d_layer_call_and_return_conditional_losses_6434u
IdentityIdentity!out01_1d/PartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:����������`�
NoOpNoOp^conv00/StatefulPartitionedCall ^conv01a/StatefulPartitionedCall ^conv01b/StatefulPartitionedCall^conv10/StatefulPartitionedCall^out01/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:����������`: : : : : : : : : : 2@
conv00/StatefulPartitionedCallconv00/StatefulPartitionedCall2B
conv01a/StatefulPartitionedCallconv01a/StatefulPartitionedCall2B
conv01b/StatefulPartitionedCallconv01b/StatefulPartitionedCall2@
conv10/StatefulPartitionedCallconv10/StatefulPartitionedCall2>
out01/StatefulPartitionedCallout01/StatefulPartitionedCall:Y U
,
_output_shapes
:����������`
%
_user_specified_nameinput_layer
�
�
&__inference_conv01b_layer_call_fn_7107

inputs!
unknown:00
	unknown_0:0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������`0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_conv01b_layer_call_and_return_conditional_losses_6392x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:����������`0`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������`0: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:����������`0
 
_user_specified_nameinputs
�

�
A__inference_conv01a_layer_call_and_return_conditional_losses_7088

inputs8
conv2d_readvariableop_resource:00-
biasadd_readvariableop_resource:0
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:00*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������`0*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������`0h
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:����������`0w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������`0: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:����������`0
 
_user_specified_nameinputs
�I
�
?__inference_model_layer_call_and_return_conditional_losses_6952

inputs?
%conv00_conv2d_readvariableop_resource:4
&conv00_biasadd_readvariableop_resource:?
%conv10_conv2d_readvariableop_resource:4
&conv10_biasadd_readvariableop_resource:@
&conv01a_conv2d_readvariableop_resource:005
'conv01a_biasadd_readvariableop_resource:0@
&conv01b_conv2d_readvariableop_resource:005
'conv01b_biasadd_readvariableop_resource:0>
$out01_conv2d_readvariableop_resource:03
%out01_biasadd_readvariableop_resource:
identity��conv00/BiasAdd/ReadVariableOp�conv00/Conv2D/ReadVariableOp�conv01a/BiasAdd/ReadVariableOp�conv01a/Conv2D/ReadVariableOp�conv01b/BiasAdd/ReadVariableOp�conv01b/Conv2D/ReadVariableOp�conv10/BiasAdd/ReadVariableOp�conv10/Conv2D/ReadVariableOp�out01/BiasAdd/ReadVariableOp�out01/Conv2D/ReadVariableOpS
reshape_1/ShapeShapeinputs*
T0*
_output_shapes
::��g
reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: i
reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
reshape_1/strided_sliceStridedSlicereshape_1/Shape:output:0&reshape_1/strided_slice/stack:output:0(reshape_1/strided_slice/stack_1:output:0(reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :�[
reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :`[
reshape_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
reshape_1/Reshape/shapePack reshape_1/strided_slice:output:0"reshape_1/Reshape/shape/1:output:0"reshape_1/Reshape/shape/2:output:0"reshape_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
reshape_1/ReshapeReshapeinputs reshape_1/Reshape/shape:output:0*
T0*0
_output_shapes
:����������`�
conv00/Conv2D/ReadVariableOpReadVariableOp%conv00_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
conv00/Conv2DConv2Dreshape_1/Reshape:output:0$conv00/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������`*
paddingSAME*
strides
�
conv00/BiasAdd/ReadVariableOpReadVariableOp&conv00_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv00/BiasAddBiasAddconv00/Conv2D:output:0%conv00/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������`w
act00/LeakyRelu	LeakyReluconv00/BiasAdd:output:0*0
_output_shapes
:����������`*
alpha%���=�
down10/MaxPoolMaxPoolact00/LeakyRelu:activations:0*/
_output_shapes
:���������@0*
ksize
*
paddingSAME*
strides
�
conv10/Conv2D/ReadVariableOpReadVariableOp%conv10_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
conv10/Conv2DConv2Ddown10/MaxPool:output:0$conv10/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@0*
paddingSAME*
strides
�
conv10/BiasAdd/ReadVariableOpReadVariableOp&conv10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv10/BiasAddBiasAddconv10/Conv2D:output:0%conv10/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@0v
act10/LeakyRelu	LeakyReluconv10/BiasAdd:output:0*/
_output_shapes
:���������@0*
alpha%���=[

up01/ConstConst*
_output_shapes
:*
dtype0*
valueB"@   0   ]
up01/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      `
up01/mulMulup01/Const:output:0up01/Const_1:output:0*
T0*
_output_shapes
:�
!up01/resize/ResizeNearestNeighborResizeNearestNeighboract10/LeakyRelu:activations:0up01/mul:z:0*
T0*0
_output_shapes
:����������`*
half_pixel_centers(V
concat01/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concat01/concatConcatV2act00/LeakyRelu:activations:02up01/resize/ResizeNearestNeighbor:resized_images:0concat01/concat/axis:output:0*
N*
T0*0
_output_shapes
:����������`0�
conv01a/Conv2D/ReadVariableOpReadVariableOp&conv01a_conv2d_readvariableop_resource*&
_output_shapes
:00*
dtype0�
conv01a/Conv2DConv2Dconcat01/concat:output:0%conv01a/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������`0*
paddingSAME*
strides
�
conv01a/BiasAdd/ReadVariableOpReadVariableOp'conv01a_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0�
conv01a/BiasAddBiasAddconv01a/Conv2D:output:0&conv01a/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������`0y
act01a/LeakyRelu	LeakyReluconv01a/BiasAdd:output:0*0
_output_shapes
:����������`0*
alpha%���=�
conv01b/Conv2D/ReadVariableOpReadVariableOp&conv01b_conv2d_readvariableop_resource*&
_output_shapes
:00*
dtype0�
conv01b/Conv2DConv2Dact01a/LeakyRelu:activations:0%conv01b/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������`0*
paddingSAME*
strides
�
conv01b/BiasAdd/ReadVariableOpReadVariableOp'conv01b_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0�
conv01b/BiasAddBiasAddconv01b/Conv2D:output:0&conv01b/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������`0y
act01b/LeakyRelu	LeakyReluconv01b/BiasAdd:output:0*0
_output_shapes
:����������`0*
alpha%���=�
out01/Conv2D/ReadVariableOpReadVariableOp$out01_conv2d_readvariableop_resource*&
_output_shapes
:0*
dtype0�
out01/Conv2DConv2Dact01b/LeakyRelu:activations:0#out01/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������`*
paddingSAME*
strides
~
out01/BiasAdd/ReadVariableOpReadVariableOp%out01_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
out01/BiasAddBiasAddout01/Conv2D:output:0$out01/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������`b
out01_1d/ShapeShapeout01/BiasAdd:output:0*
T0*
_output_shapes
::��f
out01_1d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: h
out01_1d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
out01_1d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
out01_1d/strided_sliceStridedSliceout01_1d/Shape:output:0%out01_1d/strided_slice/stack:output:0'out01_1d/strided_slice/stack_1:output:0'out01_1d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask[
out01_1d/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :�Z
out01_1d/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :`�
out01_1d/Reshape/shapePackout01_1d/strided_slice:output:0!out01_1d/Reshape/shape/1:output:0!out01_1d/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:�
out01_1d/ReshapeReshapeout01/BiasAdd:output:0out01_1d/Reshape/shape:output:0*
T0*,
_output_shapes
:����������`m
IdentityIdentityout01_1d/Reshape:output:0^NoOp*
T0*,
_output_shapes
:����������`�
NoOpNoOp^conv00/BiasAdd/ReadVariableOp^conv00/Conv2D/ReadVariableOp^conv01a/BiasAdd/ReadVariableOp^conv01a/Conv2D/ReadVariableOp^conv01b/BiasAdd/ReadVariableOp^conv01b/Conv2D/ReadVariableOp^conv10/BiasAdd/ReadVariableOp^conv10/Conv2D/ReadVariableOp^out01/BiasAdd/ReadVariableOp^out01/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:����������`: : : : : : : : : : 2>
conv00/BiasAdd/ReadVariableOpconv00/BiasAdd/ReadVariableOp2<
conv00/Conv2D/ReadVariableOpconv00/Conv2D/ReadVariableOp2@
conv01a/BiasAdd/ReadVariableOpconv01a/BiasAdd/ReadVariableOp2>
conv01a/Conv2D/ReadVariableOpconv01a/Conv2D/ReadVariableOp2@
conv01b/BiasAdd/ReadVariableOpconv01b/BiasAdd/ReadVariableOp2>
conv01b/Conv2D/ReadVariableOpconv01b/Conv2D/ReadVariableOp2>
conv10/BiasAdd/ReadVariableOpconv10/BiasAdd/ReadVariableOp2<
conv10/Conv2D/ReadVariableOpconv10/Conv2D/ReadVariableOp2<
out01/BiasAdd/ReadVariableOpout01/BiasAdd/ReadVariableOp2:
out01/Conv2D/ReadVariableOpout01/Conv2D/ReadVariableOp:T P
,
_output_shapes
:����������`
 
_user_specified_nameinputs
�

�
?__inference_out01_layer_call_and_return_conditional_losses_6415

inputs8
conv2d_readvariableop_resource:0-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:0*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������`*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������`h
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:����������`w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������`0: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:����������`0
 
_user_specified_nameinputs
�

�
@__inference_conv10_layer_call_and_return_conditional_losses_7029

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@0*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@0g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:���������@0w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@0: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������@0
 
_user_specified_nameinputs
�

�
$__inference_model_layer_call_fn_6539
input_layer!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3:00
	unknown_4:0#
	unknown_5:00
	unknown_6:0#
	unknown_7:0
	unknown_8:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_layerunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������`*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8� *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_6516t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:����������``
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:����������`: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
,
_output_shapes
:����������`
%
_user_specified_nameinput_layer
�
\
@__inference_down10_layer_call_and_return_conditional_losses_7010

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingSAME*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
\
@__inference_act01b_layer_call_and_return_conditional_losses_6403

inputs
identity`
	LeakyRelu	LeakyReluinputs*0
_output_shapes
:����������`0*
alpha%���=h
IdentityIdentityLeakyRelu:activations:0*
T0*0
_output_shapes
:����������`0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������`0:X T
0
_output_shapes
:����������`0
 
_user_specified_nameinputs
�4
�
?__inference_model_layer_call_and_return_conditional_losses_6579

inputs%
conv00_6545:
conv00_6547:%
conv10_6552:
conv10_6554:&
conv01a_6560:00
conv01a_6562:0&
conv01b_6566:00
conv01b_6568:0$

out01_6572:0

out01_6574:
identity��conv00/StatefulPartitionedCall�conv01a/StatefulPartitionedCall�conv01b/StatefulPartitionedCall�conv10/StatefulPartitionedCall�out01/StatefulPartitionedCall�
reshape_1/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_reshape_1_layer_call_and_return_conditional_losses_6300�
conv00/StatefulPartitionedCallStatefulPartitionedCall"reshape_1/PartitionedCall:output:0conv00_6545conv00_6547*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������`*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_conv00_layer_call_and_return_conditional_losses_6312�
act00/PartitionedCallPartitionedCall'conv00/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *H
fCRA
?__inference_act00_layer_call_and_return_conditional_losses_6323�
down10/PartitionedCallPartitionedCallact00/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@0* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_down10_layer_call_and_return_conditional_losses_6257�
conv10/StatefulPartitionedCallStatefulPartitionedCalldown10/PartitionedCall:output:0conv10_6552conv10_6554*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_conv10_layer_call_and_return_conditional_losses_6336�
act10/PartitionedCallPartitionedCall'conv10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@0* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *H
fCRA
?__inference_act10_layer_call_and_return_conditional_losses_6347�
up01/PartitionedCallPartitionedCallact10/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *G
fBR@
>__inference_up01_layer_call_and_return_conditional_losses_6276�
concat01/PartitionedCallPartitionedCallact00/PartitionedCall:output:0up01/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������`0* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_concat01_layer_call_and_return_conditional_losses_6357�
conv01a/StatefulPartitionedCallStatefulPartitionedCall!concat01/PartitionedCall:output:0conv01a_6560conv01a_6562*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������`0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_conv01a_layer_call_and_return_conditional_losses_6369�
act01a/PartitionedCallPartitionedCall(conv01a/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������`0* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_act01a_layer_call_and_return_conditional_losses_6380�
conv01b/StatefulPartitionedCallStatefulPartitionedCallact01a/PartitionedCall:output:0conv01b_6566conv01b_6568*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������`0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_conv01b_layer_call_and_return_conditional_losses_6392�
act01b/PartitionedCallPartitionedCall(conv01b/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������`0* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_act01b_layer_call_and_return_conditional_losses_6403�
out01/StatefulPartitionedCallStatefulPartitionedCallact01b/PartitionedCall:output:0
out01_6572
out01_6574*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������`*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *H
fCRA
?__inference_out01_layer_call_and_return_conditional_losses_6415�
out01_1d/PartitionedCallPartitionedCall&out01/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_out01_1d_layer_call_and_return_conditional_losses_6434u
IdentityIdentity!out01_1d/PartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:����������`�
NoOpNoOp^conv00/StatefulPartitionedCall ^conv01a/StatefulPartitionedCall ^conv01b/StatefulPartitionedCall^conv10/StatefulPartitionedCall^out01/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:����������`: : : : : : : : : : 2@
conv00/StatefulPartitionedCallconv00/StatefulPartitionedCall2B
conv01a/StatefulPartitionedCallconv01a/StatefulPartitionedCall2B
conv01b/StatefulPartitionedCallconv01b/StatefulPartitionedCall2@
conv10/StatefulPartitionedCallconv10/StatefulPartitionedCall2>
out01/StatefulPartitionedCallout01/StatefulPartitionedCall:T P
,
_output_shapes
:����������`
 
_user_specified_nameinputs
�
l
B__inference_concat01_layer_call_and_return_conditional_losses_6357

inputs
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :~
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*0
_output_shapes
:����������`0`
IdentityIdentityconcat:output:0*
T0*0
_output_shapes
:����������`0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:����������`:+���������������������������:X T
0
_output_shapes
:����������`
 
_user_specified_nameinputs:ie
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
[
?__inference_act00_layer_call_and_return_conditional_losses_7000

inputs
identity`
	LeakyRelu	LeakyReluinputs*0
_output_shapes
:����������`*
alpha%���=h
IdentityIdentityLeakyRelu:activations:0*
T0*0
_output_shapes
:����������`"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������`:X T
0
_output_shapes
:����������`
 
_user_specified_nameinputs
�

�
A__inference_conv01b_layer_call_and_return_conditional_losses_6392

inputs8
conv2d_readvariableop_resource:00-
biasadd_readvariableop_resource:0
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:00*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������`0*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������`0h
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:����������`0w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������`0: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:����������`0
 
_user_specified_nameinputs
�
�
$__inference_out01_layer_call_fn_7136

inputs!
unknown:0
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������`*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *H
fCRA
?__inference_out01_layer_call_and_return_conditional_losses_6415x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:����������``
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������`0: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:����������`0
 
_user_specified_nameinputs
�
A
%__inference_act01a_layer_call_fn_7093

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������`0* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_act01a_layer_call_and_return_conditional_losses_6380i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:����������`0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������`0:X T
0
_output_shapes
:����������`0
 
_user_specified_nameinputs
�
�
%__inference_conv00_layer_call_fn_6980

inputs!
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������`*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_conv00_layer_call_and_return_conditional_losses_6312x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:����������``
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������`: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:����������`
 
_user_specified_nameinputs
�4
�
?__inference_model_layer_call_and_return_conditional_losses_6516

inputs%
conv00_6482:
conv00_6484:%
conv10_6489:
conv10_6491:&
conv01a_6497:00
conv01a_6499:0&
conv01b_6503:00
conv01b_6505:0$

out01_6509:0

out01_6511:
identity��conv00/StatefulPartitionedCall�conv01a/StatefulPartitionedCall�conv01b/StatefulPartitionedCall�conv10/StatefulPartitionedCall�out01/StatefulPartitionedCall�
reshape_1/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_reshape_1_layer_call_and_return_conditional_losses_6300�
conv00/StatefulPartitionedCallStatefulPartitionedCall"reshape_1/PartitionedCall:output:0conv00_6482conv00_6484*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������`*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_conv00_layer_call_and_return_conditional_losses_6312�
act00/PartitionedCallPartitionedCall'conv00/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *H
fCRA
?__inference_act00_layer_call_and_return_conditional_losses_6323�
down10/PartitionedCallPartitionedCallact00/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@0* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_down10_layer_call_and_return_conditional_losses_6257�
conv10/StatefulPartitionedCallStatefulPartitionedCalldown10/PartitionedCall:output:0conv10_6489conv10_6491*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_conv10_layer_call_and_return_conditional_losses_6336�
act10/PartitionedCallPartitionedCall'conv10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@0* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *H
fCRA
?__inference_act10_layer_call_and_return_conditional_losses_6347�
up01/PartitionedCallPartitionedCallact10/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *G
fBR@
>__inference_up01_layer_call_and_return_conditional_losses_6276�
concat01/PartitionedCallPartitionedCallact00/PartitionedCall:output:0up01/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������`0* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_concat01_layer_call_and_return_conditional_losses_6357�
conv01a/StatefulPartitionedCallStatefulPartitionedCall!concat01/PartitionedCall:output:0conv01a_6497conv01a_6499*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������`0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_conv01a_layer_call_and_return_conditional_losses_6369�
act01a/PartitionedCallPartitionedCall(conv01a/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������`0* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_act01a_layer_call_and_return_conditional_losses_6380�
conv01b/StatefulPartitionedCallStatefulPartitionedCallact01a/PartitionedCall:output:0conv01b_6503conv01b_6505*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������`0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_conv01b_layer_call_and_return_conditional_losses_6392�
act01b/PartitionedCallPartitionedCall(conv01b/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������`0* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_act01b_layer_call_and_return_conditional_losses_6403�
out01/StatefulPartitionedCallStatefulPartitionedCallact01b/PartitionedCall:output:0
out01_6509
out01_6511*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������`*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *H
fCRA
?__inference_out01_layer_call_and_return_conditional_losses_6415�
out01_1d/PartitionedCallPartitionedCall&out01/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_out01_1d_layer_call_and_return_conditional_losses_6434u
IdentityIdentity!out01_1d/PartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:����������`�
NoOpNoOp^conv00/StatefulPartitionedCall ^conv01a/StatefulPartitionedCall ^conv01b/StatefulPartitionedCall^conv10/StatefulPartitionedCall^out01/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:����������`: : : : : : : : : : 2@
conv00/StatefulPartitionedCallconv00/StatefulPartitionedCall2B
conv01a/StatefulPartitionedCallconv01a/StatefulPartitionedCall2B
conv01b/StatefulPartitionedCallconv01b/StatefulPartitionedCall2@
conv10/StatefulPartitionedCallconv10/StatefulPartitionedCall2>
out01/StatefulPartitionedCallout01/StatefulPartitionedCall:T P
,
_output_shapes
:����������`
 
_user_specified_nameinputs
�

�
$__inference_model_layer_call_fn_6824

inputs!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3:00
	unknown_4:0#
	unknown_5:00
	unknown_6:0#
	unknown_7:0
	unknown_8:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������`*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8� *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_6579t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:����������``
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:����������`: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:����������`
 
_user_specified_nameinputs
�
[
?__inference_act10_layer_call_and_return_conditional_losses_6347

inputs
identity_
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:���������@0*
alpha%���=g
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:���������@0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@0:W S
/
_output_shapes
:���������@0
 
_user_specified_nameinputs
�4
�
?__inference_model_layer_call_and_return_conditional_losses_6475
input_layer%
conv00_6441:
conv00_6443:%
conv10_6448:
conv10_6450:&
conv01a_6456:00
conv01a_6458:0&
conv01b_6462:00
conv01b_6464:0$

out01_6468:0

out01_6470:
identity��conv00/StatefulPartitionedCall�conv01a/StatefulPartitionedCall�conv01b/StatefulPartitionedCall�conv10/StatefulPartitionedCall�out01/StatefulPartitionedCall�
reshape_1/PartitionedCallPartitionedCallinput_layer*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_reshape_1_layer_call_and_return_conditional_losses_6300�
conv00/StatefulPartitionedCallStatefulPartitionedCall"reshape_1/PartitionedCall:output:0conv00_6441conv00_6443*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������`*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_conv00_layer_call_and_return_conditional_losses_6312�
act00/PartitionedCallPartitionedCall'conv00/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *H
fCRA
?__inference_act00_layer_call_and_return_conditional_losses_6323�
down10/PartitionedCallPartitionedCallact00/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@0* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_down10_layer_call_and_return_conditional_losses_6257�
conv10/StatefulPartitionedCallStatefulPartitionedCalldown10/PartitionedCall:output:0conv10_6448conv10_6450*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_conv10_layer_call_and_return_conditional_losses_6336�
act10/PartitionedCallPartitionedCall'conv10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@0* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *H
fCRA
?__inference_act10_layer_call_and_return_conditional_losses_6347�
up01/PartitionedCallPartitionedCallact10/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *G
fBR@
>__inference_up01_layer_call_and_return_conditional_losses_6276�
concat01/PartitionedCallPartitionedCallact00/PartitionedCall:output:0up01/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������`0* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_concat01_layer_call_and_return_conditional_losses_6357�
conv01a/StatefulPartitionedCallStatefulPartitionedCall!concat01/PartitionedCall:output:0conv01a_6456conv01a_6458*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������`0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_conv01a_layer_call_and_return_conditional_losses_6369�
act01a/PartitionedCallPartitionedCall(conv01a/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������`0* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_act01a_layer_call_and_return_conditional_losses_6380�
conv01b/StatefulPartitionedCallStatefulPartitionedCallact01a/PartitionedCall:output:0conv01b_6462conv01b_6464*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������`0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_conv01b_layer_call_and_return_conditional_losses_6392�
act01b/PartitionedCallPartitionedCall(conv01b/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������`0* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_act01b_layer_call_and_return_conditional_losses_6403�
out01/StatefulPartitionedCallStatefulPartitionedCallact01b/PartitionedCall:output:0
out01_6468
out01_6470*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������`*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *H
fCRA
?__inference_out01_layer_call_and_return_conditional_losses_6415�
out01_1d/PartitionedCallPartitionedCall&out01/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_out01_1d_layer_call_and_return_conditional_losses_6434u
IdentityIdentity!out01_1d/PartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:����������`�
NoOpNoOp^conv00/StatefulPartitionedCall ^conv01a/StatefulPartitionedCall ^conv01b/StatefulPartitionedCall^conv10/StatefulPartitionedCall^out01/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:����������`: : : : : : : : : : 2@
conv00/StatefulPartitionedCallconv00/StatefulPartitionedCall2B
conv01a/StatefulPartitionedCallconv01a/StatefulPartitionedCall2B
conv01b/StatefulPartitionedCallconv01b/StatefulPartitionedCall2@
conv10/StatefulPartitionedCallconv10/StatefulPartitionedCall2>
out01/StatefulPartitionedCallout01/StatefulPartitionedCall:Y U
,
_output_shapes
:����������`
%
_user_specified_nameinput_layer
�
@
$__inference_act00_layer_call_fn_6995

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *H
fCRA
?__inference_act00_layer_call_and_return_conditional_losses_6323i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:����������`"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������`:X T
0
_output_shapes
:����������`
 
_user_specified_nameinputs
�m
�
__inference__traced_save_7271
file_prefix>
$read_disablecopyonread_conv00_kernel:2
$read_1_disablecopyonread_conv00_bias:@
&read_2_disablecopyonread_conv10_kernel:2
$read_3_disablecopyonread_conv10_bias:A
'read_4_disablecopyonread_conv01a_kernel:003
%read_5_disablecopyonread_conv01a_bias:0A
'read_6_disablecopyonread_conv01b_kernel:003
%read_7_disablecopyonread_conv01b_bias:0?
%read_8_disablecopyonread_out01_kernel:01
#read_9_disablecopyonread_out01_bias:-
#read_10_disablecopyonread_iteration:	 1
'read_11_disablecopyonread_learning_rate: )
read_12_disablecopyonread_total: )
read_13_disablecopyonread_count: 
savev2_const
identity_29��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_10/DisableCopyOnRead�Read_10/ReadVariableOp�Read_11/DisableCopyOnRead�Read_11/ReadVariableOp�Read_12/DisableCopyOnRead�Read_12/ReadVariableOp�Read_13/DisableCopyOnRead�Read_13/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOp�Read_3/DisableCopyOnRead�Read_3/ReadVariableOp�Read_4/DisableCopyOnRead�Read_4/ReadVariableOp�Read_5/DisableCopyOnRead�Read_5/ReadVariableOp�Read_6/DisableCopyOnRead�Read_6/ReadVariableOp�Read_7/DisableCopyOnRead�Read_7/ReadVariableOp�Read_8/DisableCopyOnRead�Read_8/ReadVariableOp�Read_9/DisableCopyOnRead�Read_9/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: v
Read/DisableCopyOnReadDisableCopyOnRead$read_disablecopyonread_conv00_kernel"/device:CPU:0*
_output_shapes
 �
Read/ReadVariableOpReadVariableOp$read_disablecopyonread_conv00_kernel^Read/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0q
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:i

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*&
_output_shapes
:x
Read_1/DisableCopyOnReadDisableCopyOnRead$read_1_disablecopyonread_conv00_bias"/device:CPU:0*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOp$read_1_disablecopyonread_conv00_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0i

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:_

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
:z
Read_2/DisableCopyOnReadDisableCopyOnRead&read_2_disablecopyonread_conv10_kernel"/device:CPU:0*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOp&read_2_disablecopyonread_conv10_kernel^Read_2/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0u

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:k

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*&
_output_shapes
:x
Read_3/DisableCopyOnReadDisableCopyOnRead$read_3_disablecopyonread_conv10_bias"/device:CPU:0*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOp$read_3_disablecopyonread_conv10_bias^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0i

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:_

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes
:{
Read_4/DisableCopyOnReadDisableCopyOnRead'read_4_disablecopyonread_conv01a_kernel"/device:CPU:0*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOp'read_4_disablecopyonread_conv01a_kernel^Read_4/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:00*
dtype0u

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:00k

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*&
_output_shapes
:00y
Read_5/DisableCopyOnReadDisableCopyOnRead%read_5_disablecopyonread_conv01a_bias"/device:CPU:0*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOp%read_5_disablecopyonread_conv01a_bias^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:0*
dtype0j
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:0a
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes
:0{
Read_6/DisableCopyOnReadDisableCopyOnRead'read_6_disablecopyonread_conv01b_kernel"/device:CPU:0*
_output_shapes
 �
Read_6/ReadVariableOpReadVariableOp'read_6_disablecopyonread_conv01b_kernel^Read_6/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:00*
dtype0v
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:00m
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*&
_output_shapes
:00y
Read_7/DisableCopyOnReadDisableCopyOnRead%read_7_disablecopyonread_conv01b_bias"/device:CPU:0*
_output_shapes
 �
Read_7/ReadVariableOpReadVariableOp%read_7_disablecopyonread_conv01b_bias^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:0*
dtype0j
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:0a
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes
:0y
Read_8/DisableCopyOnReadDisableCopyOnRead%read_8_disablecopyonread_out01_kernel"/device:CPU:0*
_output_shapes
 �
Read_8/ReadVariableOpReadVariableOp%read_8_disablecopyonread_out01_kernel^Read_8/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:0*
dtype0v
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:0m
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*&
_output_shapes
:0w
Read_9/DisableCopyOnReadDisableCopyOnRead#read_9_disablecopyonread_out01_bias"/device:CPU:0*
_output_shapes
 �
Read_9/ReadVariableOpReadVariableOp#read_9_disablecopyonread_out01_bias^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0j
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes
:x
Read_10/DisableCopyOnReadDisableCopyOnRead#read_10_disablecopyonread_iteration"/device:CPU:0*
_output_shapes
 �
Read_10/ReadVariableOpReadVariableOp#read_10_disablecopyonread_iteration^Read_10/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0	g
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
: ]
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0	*
_output_shapes
: |
Read_11/DisableCopyOnReadDisableCopyOnRead'read_11_disablecopyonread_learning_rate"/device:CPU:0*
_output_shapes
 �
Read_11/ReadVariableOpReadVariableOp'read_11_disablecopyonread_learning_rate^Read_11/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_12/DisableCopyOnReadDisableCopyOnReadread_12_disablecopyonread_total"/device:CPU:0*
_output_shapes
 �
Read_12/ReadVariableOpReadVariableOpread_12_disablecopyonread_total^Read_12/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_13/DisableCopyOnReadDisableCopyOnReadread_13_disablecopyonread_count"/device:CPU:0*
_output_shapes
 �
Read_13/ReadVariableOpReadVariableOpread_13_disablecopyonread_count^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtypes
2	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 i
Identity_28Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: U
Identity_29IdentityIdentity_28:output:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "#
identity_29Identity_29:output:0*3
_input_shapes"
 : : : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp26
Read_12/DisableCopyOnReadRead_12/DisableCopyOnRead20
Read_12/ReadVariableOpRead_12/ReadVariableOp26
Read_13/DisableCopyOnReadRead_13/DisableCopyOnRead20
Read_13/ReadVariableOpRead_13/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: 
�>
�
 __inference__traced_restore_7323
file_prefix8
assignvariableop_conv00_kernel:,
assignvariableop_1_conv00_bias::
 assignvariableop_2_conv10_kernel:,
assignvariableop_3_conv10_bias:;
!assignvariableop_4_conv01a_kernel:00-
assignvariableop_5_conv01a_bias:0;
!assignvariableop_6_conv01b_kernel:00-
assignvariableop_7_conv01b_bias:09
assignvariableop_8_out01_kernel:0+
assignvariableop_9_out01_bias:'
assignvariableop_10_iteration:	 +
!assignvariableop_11_learning_rate: #
assignvariableop_12_total: #
assignvariableop_13_count: 
identity_15��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_2�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*P
_output_shapes>
<:::::::::::::::*
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOpassignvariableop_conv00_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv00_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp assignvariableop_2_conv10_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOpassignvariableop_3_conv10_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp!assignvariableop_4_conv01a_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOpassignvariableop_5_conv01a_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp!assignvariableop_6_conv01b_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOpassignvariableop_7_conv01b_biasIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOpassignvariableop_8_out01_kernelIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOpassignvariableop_9_out01_biasIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_10AssignVariableOpassignvariableop_10_iterationIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp!assignvariableop_11_learning_rateIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOpassignvariableop_12_totalIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOpassignvariableop_13_countIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Identity_14Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_15IdentityIdentity_14:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_15Identity_15:output:0*1
_input_shapes 
: : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�

�
A__inference_conv01b_layer_call_and_return_conditional_losses_7117

inputs8
conv2d_readvariableop_resource:00-
biasadd_readvariableop_resource:0
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:00*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������`0*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������`0h
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:����������`0w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������`0: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:����������`0
 
_user_specified_nameinputs
�

�
?__inference_out01_layer_call_and_return_conditional_losses_7146

inputs8
conv2d_readvariableop_resource:0-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:0*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������`*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������`h
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:����������`w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������`0: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:����������`0
 
_user_specified_nameinputs
�
\
@__inference_act01b_layer_call_and_return_conditional_losses_7127

inputs
identity`
	LeakyRelu	LeakyReluinputs*0
_output_shapes
:����������`0*
alpha%���=h
IdentityIdentityLeakyRelu:activations:0*
T0*0
_output_shapes
:����������`0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������`0:X T
0
_output_shapes
:����������`0
 
_user_specified_nameinputs
�

�
@__inference_conv00_layer_call_and_return_conditional_losses_6312

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������`*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������`h
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:����������`w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������`: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:����������`
 
_user_specified_nameinputs
�
[
?__inference_act10_layer_call_and_return_conditional_losses_7039

inputs
identity_
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:���������@0*
alpha%���=g
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:���������@0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@0:W S
/
_output_shapes
:���������@0
 
_user_specified_nameinputs
�
A
%__inference_down10_layer_call_fn_7005

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_down10_layer_call_and_return_conditional_losses_6257�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
_
C__inference_reshape_1_layer_call_and_return_conditional_losses_6971

inputs
identityI
ShapeShapeinputs*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskR
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :�Q
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :`Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:m
ReshapeReshapeinputsReshape/shape:output:0*
T0*0
_output_shapes
:����������`a
IdentityIdentityReshape:output:0*
T0*0
_output_shapes
:����������`"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������`:T P
,
_output_shapes
:����������`
 
_user_specified_nameinputs
�

�
@__inference_conv10_layer_call_and_return_conditional_losses_6336

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@0*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@0g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:���������@0w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@0: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������@0
 
_user_specified_nameinputs
�
S
'__inference_concat01_layer_call_fn_7062
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������`0* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_concat01_layer_call_and_return_conditional_losses_6357i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:����������`0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:����������`:+���������������������������:Z V
0
_output_shapes
:����������`
"
_user_specified_name
inputs_0:kg
A
_output_shapes/
-:+���������������������������
"
_user_specified_name
inputs_1
�

�
A__inference_conv01a_layer_call_and_return_conditional_losses_6369

inputs8
conv2d_readvariableop_resource:00-
biasadd_readvariableop_resource:0
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:00*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������`0*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������`0h
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:����������`0w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������`0: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:����������`0
 
_user_specified_nameinputs
�
@
$__inference_act10_layer_call_fn_7034

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@0* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *H
fCRA
?__inference_act10_layer_call_and_return_conditional_losses_6347h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������@0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@0:W S
/
_output_shapes
:���������@0
 
_user_specified_nameinputs
�
�
&__inference_conv01a_layer_call_fn_7078

inputs!
unknown:00
	unknown_0:0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������`0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_conv01a_layer_call_and_return_conditional_losses_6369x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:����������`0`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������`0: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:����������`0
 
_user_specified_nameinputs
�
_
C__inference_reshape_1_layer_call_and_return_conditional_losses_6300

inputs
identityI
ShapeShapeinputs*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskR
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :�Q
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :`Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:m
ReshapeReshapeinputsReshape/shape:output:0*
T0*0
_output_shapes
:����������`a
IdentityIdentityReshape:output:0*
T0*0
_output_shapes
:����������`"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������`:T P
,
_output_shapes
:����������`
 
_user_specified_nameinputs
�
Z
>__inference_up01_layer_call_and_return_conditional_losses_6276

inputs
identityI
ShapeShapeinputs*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:V
ConstConst*
_output_shapes
:*
dtype0*
valueB"      W
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:�
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4������������������������������������*
half_pixel_centers(�
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
n
B__inference_concat01_layer_call_and_return_conditional_losses_7069
inputs_0
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*0
_output_shapes
:����������`0`
IdentityIdentityconcat:output:0*
T0*0
_output_shapes
:����������`0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:����������`:+���������������������������:Z V
0
_output_shapes
:����������`
"
_user_specified_name
inputs_0:kg
A
_output_shapes/
-:+���������������������������
"
_user_specified_name
inputs_1
�
\
@__inference_down10_layer_call_and_return_conditional_losses_6257

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingSAME*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
?
#__inference_up01_layer_call_fn_7044

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *G
fBR@
>__inference_up01_layer_call_and_return_conditional_losses_6276�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
\
@__inference_act01a_layer_call_and_return_conditional_losses_7098

inputs
identity`
	LeakyRelu	LeakyReluinputs*0
_output_shapes
:����������`0*
alpha%���=h
IdentityIdentityLeakyRelu:activations:0*
T0*0
_output_shapes
:����������`0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������`0:X T
0
_output_shapes
:����������`0
 
_user_specified_nameinputs
�
A
%__inference_act01b_layer_call_fn_7122

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������`0* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_act01b_layer_call_and_return_conditional_losses_6403i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:����������`0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������`0:X T
0
_output_shapes
:����������`0
 
_user_specified_nameinputs
�
�
%__inference_conv10_layer_call_fn_7019

inputs!
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_conv10_layer_call_and_return_conditional_losses_6336w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������@0`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@0: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������@0
 
_user_specified_nameinputs
�
D
(__inference_reshape_1_layer_call_fn_6957

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_reshape_1_layer_call_and_return_conditional_losses_6300i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:����������`"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������`:T P
,
_output_shapes
:����������`
 
_user_specified_nameinputs
�

^
B__inference_out01_1d_layer_call_and_return_conditional_losses_7164

inputs
identityI
ShapeShapeinputs*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskR
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :�Q
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :`�
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:i
ReshapeReshapeinputsReshape/shape:output:0*
T0*,
_output_shapes
:����������`]
IdentityIdentityReshape:output:0*
T0*,
_output_shapes
:����������`"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������`:X T
0
_output_shapes
:����������`
 
_user_specified_nameinputs
�

�
$__inference_model_layer_call_fn_6602
input_layer!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3:00
	unknown_4:0#
	unknown_5:00
	unknown_6:0#
	unknown_7:0
	unknown_8:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_layerunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������`*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8� *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_6579t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:����������``
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:����������`: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
,
_output_shapes
:����������`
%
_user_specified_nameinput_layer
�

�
@__inference_conv00_layer_call_and_return_conditional_losses_6990

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������`*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������`h
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:����������`w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������`: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:����������`
 
_user_specified_nameinputs
�

^
B__inference_out01_1d_layer_call_and_return_conditional_losses_6434

inputs
identityI
ShapeShapeinputs*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskR
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :�Q
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :`�
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:i
ReshapeReshapeinputsReshape/shape:output:0*
T0*,
_output_shapes
:����������`]
IdentityIdentityReshape:output:0*
T0*,
_output_shapes
:����������`"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������`:X T
0
_output_shapes
:����������`
 
_user_specified_nameinputs
�

�
$__inference_model_layer_call_fn_6799

inputs!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3:00
	unknown_4:0#
	unknown_5:00
	unknown_6:0#
	unknown_7:0
	unknown_8:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������`*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8� *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_6516t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:����������``
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:����������`: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:����������`
 
_user_specified_nameinputs
�

�
"__inference_signature_wrapper_6774
input_layer!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3:00
	unknown_4:0#
	unknown_5:00
	unknown_6:0#
	unknown_7:0
	unknown_8:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_layerunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������`*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8� *(
f#R!
__inference__wrapped_model_6251t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:����������``
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:����������`: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
,
_output_shapes
:����������`
%
_user_specified_nameinput_layer
�
[
?__inference_act00_layer_call_and_return_conditional_losses_6323

inputs
identity`
	LeakyRelu	LeakyReluinputs*0
_output_shapes
:����������`*
alpha%���=h
IdentityIdentityLeakyRelu:activations:0*
T0*0
_output_shapes
:����������`"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������`:X T
0
_output_shapes
:����������`
 
_user_specified_nameinputs"�
L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
H
input_layer9
serving_default_input_layer:0����������`A
out01_1d5
StatefulPartitionedCall:0����������`tensorflow/serving/predict:��
�
layer-0
layer-1
layer_with_weights-0
layer-2
layer-3
layer-4
layer_with_weights-1
layer-5
layer-6
layer-7
	layer-8

layer_with_weights-2

layer-9
layer-10
layer_with_weights-3
layer-11
layer-12
layer_with_weights-4
layer-13
layer-14
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_network
"
_tf_keras_input_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
�
	variables
 trainable_variables
!regularization_losses
"	keras_api
#__call__
*$&call_and_return_all_conditional_losses

%kernel
&bias
 '_jit_compiled_convolution_op"
_tf_keras_layer
�
(	variables
)trainable_variables
*regularization_losses
+	keras_api
,__call__
*-&call_and_return_all_conditional_losses"
_tf_keras_layer
�
.	variables
/trainable_variables
0regularization_losses
1	keras_api
2__call__
*3&call_and_return_all_conditional_losses"
_tf_keras_layer
�
4	variables
5trainable_variables
6regularization_losses
7	keras_api
8__call__
*9&call_and_return_all_conditional_losses

:kernel
;bias
 <_jit_compiled_convolution_op"
_tf_keras_layer
�
=	variables
>trainable_variables
?regularization_losses
@	keras_api
A__call__
*B&call_and_return_all_conditional_losses"
_tf_keras_layer
�
C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
G__call__
*H&call_and_return_all_conditional_losses"
_tf_keras_layer
�
I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
M__call__
*N&call_and_return_all_conditional_losses"
_tf_keras_layer
�
O	variables
Ptrainable_variables
Qregularization_losses
R	keras_api
S__call__
*T&call_and_return_all_conditional_losses

Ukernel
Vbias
 W_jit_compiled_convolution_op"
_tf_keras_layer
�
X	variables
Ytrainable_variables
Zregularization_losses
[	keras_api
\__call__
*]&call_and_return_all_conditional_losses"
_tf_keras_layer
�
^	variables
_trainable_variables
`regularization_losses
a	keras_api
b__call__
*c&call_and_return_all_conditional_losses

dkernel
ebias
 f_jit_compiled_convolution_op"
_tf_keras_layer
�
g	variables
htrainable_variables
iregularization_losses
j	keras_api
k__call__
*l&call_and_return_all_conditional_losses"
_tf_keras_layer
�
m	variables
ntrainable_variables
oregularization_losses
p	keras_api
q__call__
*r&call_and_return_all_conditional_losses

skernel
tbias
 u_jit_compiled_convolution_op"
_tf_keras_layer
�
v	variables
wtrainable_variables
xregularization_losses
y	keras_api
z__call__
*{&call_and_return_all_conditional_losses"
_tf_keras_layer
f
%0
&1
:2
;3
U4
V5
d6
e7
s8
t9"
trackable_list_wrapper
f
%0
&1
:2
;3
U4
V5
d6
e7
s8
t9"
trackable_list_wrapper
 "
trackable_list_wrapper
�
|non_trainable_variables

}layers
~metrics
layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_1
�trace_2
�trace_32�
$__inference_model_layer_call_fn_6539
$__inference_model_layer_call_fn_6602
$__inference_model_layer_call_fn_6799
$__inference_model_layer_call_fn_6824�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1z�trace_2z�trace_3
�
�trace_0
�trace_1
�trace_2
�trace_32�
?__inference_model_layer_call_and_return_conditional_losses_6437
?__inference_model_layer_call_and_return_conditional_losses_6475
?__inference_model_layer_call_and_return_conditional_losses_6888
?__inference_model_layer_call_and_return_conditional_losses_6952�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1z�trace_2z�trace_3
�B�
__inference__wrapped_model_6251input_layer"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
n
�
_variables
�_iterations
�_learning_rate
�_update_step_xla"
experimentalOptimizer
-
�serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_reshape_1_layer_call_fn_6957�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
C__inference_reshape_1_layer_call_and_return_conditional_losses_6971�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
.
%0
&1"
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
 trainable_variables
!regularization_losses
#__call__
*$&call_and_return_all_conditional_losses
&$"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
%__inference_conv00_layer_call_fn_6980�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
@__inference_conv00_layer_call_and_return_conditional_losses_6990�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
':%2conv00/kernel
:2conv00/bias
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
(	variables
)trainable_variables
*regularization_losses
,__call__
*-&call_and_return_all_conditional_losses
&-"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
$__inference_act00_layer_call_fn_6995�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
?__inference_act00_layer_call_and_return_conditional_losses_7000�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
.	variables
/trainable_variables
0regularization_losses
2__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
%__inference_down10_layer_call_fn_7005�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
@__inference_down10_layer_call_and_return_conditional_losses_7010�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
.
:0
;1"
trackable_list_wrapper
.
:0
;1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
4	variables
5trainable_variables
6regularization_losses
8__call__
*9&call_and_return_all_conditional_losses
&9"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
%__inference_conv10_layer_call_fn_7019�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
@__inference_conv10_layer_call_and_return_conditional_losses_7029�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
':%2conv10/kernel
:2conv10/bias
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
=	variables
>trainable_variables
?regularization_losses
A__call__
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
$__inference_act10_layer_call_fn_7034�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
?__inference_act10_layer_call_and_return_conditional_losses_7039�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
C	variables
Dtrainable_variables
Eregularization_losses
G__call__
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
#__inference_up01_layer_call_fn_7044�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
>__inference_up01_layer_call_and_return_conditional_losses_7056�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
I	variables
Jtrainable_variables
Kregularization_losses
M__call__
*N&call_and_return_all_conditional_losses
&N"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
'__inference_concat01_layer_call_fn_7062�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
B__inference_concat01_layer_call_and_return_conditional_losses_7069�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
.
U0
V1"
trackable_list_wrapper
.
U0
V1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
O	variables
Ptrainable_variables
Qregularization_losses
S__call__
*T&call_and_return_all_conditional_losses
&T"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
&__inference_conv01a_layer_call_fn_7078�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
A__inference_conv01a_layer_call_and_return_conditional_losses_7088�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
(:&002conv01a/kernel
:02conv01a/bias
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
X	variables
Ytrainable_variables
Zregularization_losses
\__call__
*]&call_and_return_all_conditional_losses
&]"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
%__inference_act01a_layer_call_fn_7093�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
@__inference_act01a_layer_call_and_return_conditional_losses_7098�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
.
d0
e1"
trackable_list_wrapper
.
d0
e1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
^	variables
_trainable_variables
`regularization_losses
b__call__
*c&call_and_return_all_conditional_losses
&c"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
&__inference_conv01b_layer_call_fn_7107�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
A__inference_conv01b_layer_call_and_return_conditional_losses_7117�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
(:&002conv01b/kernel
:02conv01b/bias
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
g	variables
htrainable_variables
iregularization_losses
k__call__
*l&call_and_return_all_conditional_losses
&l"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
%__inference_act01b_layer_call_fn_7122�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
@__inference_act01b_layer_call_and_return_conditional_losses_7127�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
.
s0
t1"
trackable_list_wrapper
.
s0
t1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
m	variables
ntrainable_variables
oregularization_losses
q__call__
*r&call_and_return_all_conditional_losses
&r"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
$__inference_out01_layer_call_fn_7136�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
?__inference_out01_layer_call_and_return_conditional_losses_7146�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
&:$02out01/kernel
:2
out01/bias
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
v	variables
wtrainable_variables
xregularization_losses
z__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
'__inference_out01_1d_layer_call_fn_7151�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
B__inference_out01_1d_layer_call_and_return_conditional_losses_7164�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
�
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
$__inference_model_layer_call_fn_6539input_layer"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
$__inference_model_layer_call_fn_6602input_layer"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
$__inference_model_layer_call_fn_6799inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
$__inference_model_layer_call_fn_6824inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
?__inference_model_layer_call_and_return_conditional_losses_6437input_layer"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
?__inference_model_layer_call_and_return_conditional_losses_6475input_layer"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
?__inference_model_layer_call_and_return_conditional_losses_6888inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
?__inference_model_layer_call_and_return_conditional_losses_6952inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
(
�0"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
�2��
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
�B�
"__inference_signature_wrapper_6774input_layer"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_reshape_1_layer_call_fn_6957inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_reshape_1_layer_call_and_return_conditional_losses_6971inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
%__inference_conv00_layer_call_fn_6980inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
@__inference_conv00_layer_call_and_return_conditional_losses_6990inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
$__inference_act00_layer_call_fn_6995inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
?__inference_act00_layer_call_and_return_conditional_losses_7000inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
%__inference_down10_layer_call_fn_7005inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
@__inference_down10_layer_call_and_return_conditional_losses_7010inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
%__inference_conv10_layer_call_fn_7019inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
@__inference_conv10_layer_call_and_return_conditional_losses_7029inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
$__inference_act10_layer_call_fn_7034inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
?__inference_act10_layer_call_and_return_conditional_losses_7039inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
#__inference_up01_layer_call_fn_7044inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
>__inference_up01_layer_call_and_return_conditional_losses_7056inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
'__inference_concat01_layer_call_fn_7062inputs_0inputs_1"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_concat01_layer_call_and_return_conditional_losses_7069inputs_0inputs_1"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
&__inference_conv01a_layer_call_fn_7078inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
A__inference_conv01a_layer_call_and_return_conditional_losses_7088inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
%__inference_act01a_layer_call_fn_7093inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
@__inference_act01a_layer_call_and_return_conditional_losses_7098inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
&__inference_conv01b_layer_call_fn_7107inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
A__inference_conv01b_layer_call_and_return_conditional_losses_7117inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
%__inference_act01b_layer_call_fn_7122inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
@__inference_act01b_layer_call_and_return_conditional_losses_7127inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
$__inference_out01_layer_call_fn_7136inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
?__inference_out01_layer_call_and_return_conditional_losses_7146inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
'__inference_out01_1d_layer_call_fn_7151inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_out01_1d_layer_call_and_return_conditional_losses_7164inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
R
�	variables
�	keras_api

�total

�count"
_tf_keras_metric
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count�
__inference__wrapped_model_6251�
%&:;UVdest9�6
/�,
*�'
input_layer����������`
� "8�5
3
out01_1d'�$
out01_1d����������`�
?__inference_act00_layer_call_and_return_conditional_losses_7000q8�5
.�+
)�&
inputs����������`
� "5�2
+�(
tensor_0����������`
� �
$__inference_act00_layer_call_fn_6995f8�5
.�+
)�&
inputs����������`
� "*�'
unknown����������`�
@__inference_act01a_layer_call_and_return_conditional_losses_7098q8�5
.�+
)�&
inputs����������`0
� "5�2
+�(
tensor_0����������`0
� �
%__inference_act01a_layer_call_fn_7093f8�5
.�+
)�&
inputs����������`0
� "*�'
unknown����������`0�
@__inference_act01b_layer_call_and_return_conditional_losses_7127q8�5
.�+
)�&
inputs����������`0
� "5�2
+�(
tensor_0����������`0
� �
%__inference_act01b_layer_call_fn_7122f8�5
.�+
)�&
inputs����������`0
� "*�'
unknown����������`0�
?__inference_act10_layer_call_and_return_conditional_losses_7039o7�4
-�*
(�%
inputs���������@0
� "4�1
*�'
tensor_0���������@0
� �
$__inference_act10_layer_call_fn_7034d7�4
-�*
(�%
inputs���������@0
� ")�&
unknown���������@0�
B__inference_concat01_layer_call_and_return_conditional_losses_7069�}�z
s�p
n�k
+�(
inputs_0����������`
<�9
inputs_1+���������������������������
� "5�2
+�(
tensor_0����������`0
� �
'__inference_concat01_layer_call_fn_7062�}�z
s�p
n�k
+�(
inputs_0����������`
<�9
inputs_1+���������������������������
� "*�'
unknown����������`0�
@__inference_conv00_layer_call_and_return_conditional_losses_6990u%&8�5
.�+
)�&
inputs����������`
� "5�2
+�(
tensor_0����������`
� �
%__inference_conv00_layer_call_fn_6980j%&8�5
.�+
)�&
inputs����������`
� "*�'
unknown����������`�
A__inference_conv01a_layer_call_and_return_conditional_losses_7088uUV8�5
.�+
)�&
inputs����������`0
� "5�2
+�(
tensor_0����������`0
� �
&__inference_conv01a_layer_call_fn_7078jUV8�5
.�+
)�&
inputs����������`0
� "*�'
unknown����������`0�
A__inference_conv01b_layer_call_and_return_conditional_losses_7117ude8�5
.�+
)�&
inputs����������`0
� "5�2
+�(
tensor_0����������`0
� �
&__inference_conv01b_layer_call_fn_7107jde8�5
.�+
)�&
inputs����������`0
� "*�'
unknown����������`0�
@__inference_conv10_layer_call_and_return_conditional_losses_7029s:;7�4
-�*
(�%
inputs���������@0
� "4�1
*�'
tensor_0���������@0
� �
%__inference_conv10_layer_call_fn_7019h:;7�4
-�*
(�%
inputs���������@0
� ")�&
unknown���������@0�
@__inference_down10_layer_call_and_return_conditional_losses_7010�R�O
H�E
C�@
inputs4������������������������������������
� "O�L
E�B
tensor_04������������������������������������
� �
%__inference_down10_layer_call_fn_7005�R�O
H�E
C�@
inputs4������������������������������������
� "D�A
unknown4�������������������������������������
?__inference_model_layer_call_and_return_conditional_losses_6437�
%&:;UVdestA�>
7�4
*�'
input_layer����������`
p

 
� "1�.
'�$
tensor_0����������`
� �
?__inference_model_layer_call_and_return_conditional_losses_6475�
%&:;UVdestA�>
7�4
*�'
input_layer����������`
p 

 
� "1�.
'�$
tensor_0����������`
� �
?__inference_model_layer_call_and_return_conditional_losses_6888}
%&:;UVdest<�9
2�/
%�"
inputs����������`
p

 
� "1�.
'�$
tensor_0����������`
� �
?__inference_model_layer_call_and_return_conditional_losses_6952}
%&:;UVdest<�9
2�/
%�"
inputs����������`
p 

 
� "1�.
'�$
tensor_0����������`
� �
$__inference_model_layer_call_fn_6539w
%&:;UVdestA�>
7�4
*�'
input_layer����������`
p

 
� "&�#
unknown����������`�
$__inference_model_layer_call_fn_6602w
%&:;UVdestA�>
7�4
*�'
input_layer����������`
p 

 
� "&�#
unknown����������`�
$__inference_model_layer_call_fn_6799r
%&:;UVdest<�9
2�/
%�"
inputs����������`
p

 
� "&�#
unknown����������`�
$__inference_model_layer_call_fn_6824r
%&:;UVdest<�9
2�/
%�"
inputs����������`
p 

 
� "&�#
unknown����������`�
B__inference_out01_1d_layer_call_and_return_conditional_losses_7164m8�5
.�+
)�&
inputs����������`
� "1�.
'�$
tensor_0����������`
� �
'__inference_out01_1d_layer_call_fn_7151b8�5
.�+
)�&
inputs����������`
� "&�#
unknown����������`�
?__inference_out01_layer_call_and_return_conditional_losses_7146ust8�5
.�+
)�&
inputs����������`0
� "5�2
+�(
tensor_0����������`
� �
$__inference_out01_layer_call_fn_7136jst8�5
.�+
)�&
inputs����������`0
� "*�'
unknown����������`�
C__inference_reshape_1_layer_call_and_return_conditional_losses_6971m4�1
*�'
%�"
inputs����������`
� "5�2
+�(
tensor_0����������`
� �
(__inference_reshape_1_layer_call_fn_6957b4�1
*�'
%�"
inputs����������`
� "*�'
unknown����������`�
"__inference_signature_wrapper_6774�
%&:;UVdestH�E
� 
>�;
9
input_layer*�'
input_layer����������`"8�5
3
out01_1d'�$
out01_1d����������`�
>__inference_up01_layer_call_and_return_conditional_losses_7056�R�O
H�E
C�@
inputs4������������������������������������
� "O�L
E�B
tensor_04������������������������������������
� �
#__inference_up01_layer_call_fn_7044�R�O
H�E
C�@
inputs4������������������������������������
� "D�A
unknown4������������������������������������