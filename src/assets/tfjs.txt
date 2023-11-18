namespace tf {
/// <amd-module name="@tensorflow/tfjs-core/dist/ops/abs" />
/**
 * Computes absolute value element-wise: `abs(x)`
 *
 * ```js
 * const x = tf.tensor1d([-1, 2, -3, 4]);
 *
 * x.abs().print();  // or tf.abs(x)
 * ```
 * @param x The input `tf.Tensor`.
 *
 * @doc {heading: 'Operations', subheading: 'Basic math'}
 */
declare function abs_<T extends Tensor>(x: T | TensorLike): T;
declare const abs: typeof abs_;
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/losses/absolute_difference" />
/**
 * Computes the absolute difference loss between two tensors.
 *
 * @param labels The ground truth output tensor, same dimensions as
 *    'predictions'.
 * @param predictions The predicted outputs.
 * @param weights Tensor whose rank is either 0, or the same rank as
 *    `labels`, and must be broadcastable to `labels` (i.e., all dimensions
 *    must be either `1`, or the same as the corresponding `losses`
 *    dimension).
 * @param reduction Type of reduction to apply to loss. Should be of type
 *    `Reduction`
 *
 * @doc {heading: 'Training', subheading: 'Losses', namespace: 'losses'}
 */
declare function absoluteDifference_<T extends Tensor, O extends Tensor>(labels: T | TensorLike, predictions: T | TensorLike, weights?: Tensor | TensorLike, reduction?: Reduction): O;
declare const absoluteDifference: typeof absoluteDifference_;
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/losses/absolute_difference_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/gradients/Abs_grad" />
declare const absGradConfig: GradConfig;

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/abs_test" />
{ };
/// <amd-module name="@tensorflow/tfjs-core/dist/ops/acos" />
/**
 * Computes acos of the input `tf.Tensor` element-wise: `acos(x)`
 *
 * ```js
 * const x = tf.tensor1d([0, 1, -1, .7]);
 *
 * x.acos().print();  // or tf.acos(x)
 * ```
 * @param x The input tensor.
 * @doc {heading: 'Operations', subheading: 'Basic math'}
 */
declare function acos_<T extends Tensor>(x: T | TensorLike): T;
declare const acos: typeof acos_;
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/public/chained_ops/acosh" />
declare module '../../tensor' {
    interface Tensor<R extends Rank = Rank> {
        acosh<T extends Tensor>(this: T): T;
    }
}

/// <amd-module name="@tensorflow/tfjs-core/dist/gradients/Acosh_grad" />
declare const acoshGradConfig: GradConfig;

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/acosh_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/gradients/Acos_grad" />
declare const acosGradConfig: GradConfig;

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/acos_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-layers/dist/activations" />
/**
 * Base class for Activations.
 *
 * Special note: due to cross-language compatibility reasons, the
 * static readonly className field in this family of classes must be set to
 * the initialLowerCamelCase name of the activation.
 */
declare abstract class Activation extends serialization.Serializable {
    abstract apply(tensor: Tensor, axis?: number): Tensor;
    getConfig(): serialization.ConfigDict;
}
/**
 * Exponential linear unit (ELU).
 * Reference: https://arxiv.org/abs/1511.07289
 */
declare class Elu extends Activation {
    /** @nocollapse */
    static readonly className = "elu";
    /**
     * Calculate the activation function.
     *
     * @param x: Input.
     * @param alpha: Scaling factor the negative section.
     * @return Output of the ELU activation.
     */
    apply(x: Tensor, alpha?: number): Tensor;
}
/**
 * Scaled Exponential Linear Unit. (Klambauer et al., 2017).
 * Reference: Self-Normalizing Neural Networks, https://arxiv.org/abs/1706.02515
 * Notes:
 *   - To be used together with the initialization "lecunNormal".
 *   - To be used together with the dropout variant "AlphaDropout".
 */
declare class Selu extends Activation {
    /** @nocollapse */
    static readonly className = "selu";
    apply(x: Tensor): Tensor;
}
/**
 *  Rectified linear unit
 */
declare class Relu extends Activation {
    /** @nocollapse */
    static readonly className = "relu";
    apply(x: Tensor): Tensor;
}
/**
 * Rectified linear unit activation maxing out at 6.0.
 */
declare class Relu6 extends Activation {
    /** @nocollapse */
    static readonly className = "relu6";
    apply(x: Tensor): Tensor;
}
declare class Linear extends Activation {
    /** @nocollapse */
    static readonly className = "linear";
    apply(x: Tensor): Tensor;
}
/**
 * Sigmoid activation function.
 */
declare class Sigmoid extends Activation {
    /** @nocollapse */
    static readonly className = "sigmoid";
    apply(x: Tensor): Tensor;
}
/**
 * Segment-wise linear approximation of sigmoid.
 */
declare class HardSigmoid extends Activation {
    /** @nocollapse */
    static readonly className = "hardSigmoid";
    apply(x: Tensor): Tensor;
}
/**
 * Softplus activation function.
 */
declare class Softplus extends Activation {
    /** @nocollapse */
    static readonly className = "softplus";
    apply(x: Tensor): Tensor;
}
/**
 * Softsign activation function.
 */
declare class Softsign extends Activation {
    /** @nocollapse */
    static readonly className = "softsign";
    apply(x: Tensor): Tensor;
}
/**
 * Hyperbolic tangent function.
 */
declare class Tanh extends Activation {
    /** @nocollapse */
    static readonly className = "tanh";
    apply(x: Tensor): Tensor;
}
/**
 * Softmax activation function
 */
declare class Softmax extends Activation {
    /** @nocollapse */
    static readonly className = "softmax";
    /**
     * Calculate the activation function.
     *
     * @param x Tensor.
     * @param axis Integer, axis along which the softmax normalization is applied.
     * Invalid if < 2, as softmax across 1 (the batch dimension) is assumed to be
     * an error.
     *
     * @returns a Tensor of the same shape as x
     *
     * @throws ValueError: In case `dim(x) < 2`.
     */
    apply(x: Tensor, axis?: number): Tensor;
}
/**
 * Log softmax activation function
 */
declare class LogSoftmax extends Activation {
    /** @nocollapse */
    static readonly className = "logSoftmax";
    /**
     * Calculate the activation function of log softmax:
     * log( exp(x_i) / sum(exp(x)) )
     *
     * @param x Tensor.
     * @param axis Integer, axis along which the softmax normalization is applied.
     * Invalid if < 2, as softmax across 1 (the batch dimension) is assumed to be
     * an error.
     *
     * @returns a Tensor of the same shape as x
     *
     * @throws ValueError: In case `dim(x) < 2`.
     */
    apply(x: Tensor, axis?: number): Tensor;
}
/**
 * Swish activation function
 */
declare class Swish extends Activation {
    /** @nocollapse */
    static readonly className = "swish";
    /**
     * Calculate the activation function.
     *
     * @param x Tensor.
     * @param alpha Scaling factor for the sigmoid function.
     * @returns a Tensor of the same shape as x
     */
    apply(x: Tensor, alpha?: number): Tensor;
}
/**
 * Mish activation function
 */
declare class Mish extends Activation {
    /** @nocollapse */
    static readonly className = "mish";
    /**
     * Calculate the activation function.
     *
     * @param x Tensor.
     * @returns a Tensor of the same shape as x
     */
    apply(x: Tensor): Tensor;
}
declare function serializeActivation(activation: Activation): string;
declare function deserializeActivation(config: serialization.ConfigDict, customObjects?: serialization.ConfigDict): Activation;
declare function getActivation(identifier: ActivationIdentifier | serialization.ConfigDict | Activation): Activation;

/// <amd-module name="@tensorflow/tfjs-layers/dist/keras_format/activation_config" />
/**
 * List of all known activation names.
 */
declare const activationOptions: ("linear" | "relu" | "elu" | "relu6" | "sigmoid" | "hard_sigmoid" | "selu" | "softmax" | "softplus" | "softsign" | "tanh" | "swish" | "mish")[];
/**
 * A type representing the strings that are valid loss names.
 */
declare type ActivationSerialization = typeof activationOptions[number];
/** @docinline */
declare type ActivationIdentifier = 'elu' | 'hardSigmoid' | 'linear' | 'relu' | 'relu6' | 'selu' | 'sigmoid' | 'softmax' | 'softplus' | 'softsign' | 'tanh' | 'swish' | 'mish';

/// <amd-module name="@tensorflow/tfjs-core/dist/optimizers/adadelta_optimizer" />
/** @doclink Optimizer */
declare class AdadeltaOptimizer extends Optimizer {
    protected learningRate: number;
    protected rho: number;
    protected epsilon: number;
    /** @nocollapse */
    static get className(): string;
    private accumulatedGrads;
    private accumulatedUpdates;
    constructor(learningRate: number, rho: number, epsilon?: number);
    applyGradients(variableGradients: NamedVariableMap | NamedTensor[]): void;
    dispose(): void;
    getWeights(): Promise<NamedTensor[]>;
    setWeights(weightValues: NamedTensor[]): Promise<void>;
    getConfig(): ConfigDict;
    /** @nocollapse */
    static fromConfig<T extends Serializable>(cls: SerializableConstructor<T>, config: ConfigDict): T;
}

/// <amd-module name="@tensorflow/tfjs-core/dist/optimizers/adadelta_optimizer_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/optimizers/adagrad_optimizer" />
/** @doclink Optimizer */
declare class AdagradOptimizer extends Optimizer {
    protected learningRate: number;
    private initialAccumulatorValue;
    /** @nocollapse */
    static get className(): string;
    private accumulatedGrads;
    constructor(learningRate: number, initialAccumulatorValue?: number);
    applyGradients(variableGradients: NamedVariableMap | NamedTensor[]): void;
    dispose(): void;
    getWeights(): Promise<NamedTensor[]>;
    setWeights(weightValues: NamedTensor[]): Promise<void>;
    getConfig(): ConfigDict;
    /** @nocollapse */
    static fromConfig<T extends Serializable>(cls: SerializableConstructor<T>, config: ConfigDict): T;
}

/// <amd-module name="@tensorflow/tfjs-core/dist/optimizers/adagrad_optimizer_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/optimizers/adamax_optimizer" />
declare class AdamaxOptimizer extends Optimizer {
    protected learningRate: number;
    protected beta1: number;
    protected beta2: number;
    protected epsilon: number;
    protected decay: number;
    /** @nocollapse */
    static get className(): string;
    private accBeta1;
    private iteration;
    private accumulatedFirstMoment;
    private accumulatedWeightedInfNorm;
    constructor(learningRate: number, beta1: number, beta2: number, epsilon?: number, decay?: number);
    applyGradients(variableGradients: NamedVariableMap | NamedTensor[]): void;
    dispose(): void;
    getWeights(): Promise<NamedTensor[]>;
    setWeights(weightValues: NamedTensor[]): Promise<void>;
    getConfig(): ConfigDict;
    /** @nocollapse */
    static fromConfig<T extends Serializable>(cls: SerializableConstructor<T>, config: ConfigDict): T;
}

/// <amd-module name="@tensorflow/tfjs-core/dist/optimizers/adamax_optimizer_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/optimizers/adam_optimizer" />
declare class AdamOptimizer extends Optimizer {
    protected learningRate: number;
    protected beta1: number;
    protected beta2: number;
    protected epsilon: number;
    /** @nocollapse */
    static get className(): string;
    private accBeta1;
    private accBeta2;
    private accumulatedFirstMoment;
    private accumulatedSecondMoment;
    constructor(learningRate: number, beta1: number, beta2: number, epsilon?: number);
    applyGradients(variableGradients: NamedVariableMap | NamedTensor[]): void;
    dispose(): void;
    getWeights(): Promise<NamedTensor[]>;
    setWeights(weightValues: NamedTensor[]): Promise<void>;
    getConfig(): ConfigDict;
    /** @nocollapse */
    static fromConfig<T extends Serializable>(cls: SerializableConstructor<T>, config: ConfigDict): T;
}

/// <amd-module name="@tensorflow/tfjs-core/dist/optimizers/adam_optimizer_test" />
{ };
/// <amd-module name="@tensorflow/tfjs-core/dist/ops/add" />
/**
 * Adds two `tf.Tensor`s element-wise, A + B. Supports broadcasting.
 *
 *
 * ```js
 * const a = tf.tensor1d([1, 2, 3, 4]);
 * const b = tf.tensor1d([10, 20, 30, 40]);
 *
 * a.add(b).print();  // or tf.add(a, b)
 * ```
 *
 * ```js
 * // Broadcast add a with b.
 * const a = tf.scalar(5);
 * const b = tf.tensor1d([10, 20, 30, 40]);
 *
 * a.add(b).print();  // or tf.add(a, b)
 * ```
 * @param a The first `tf.Tensor` to add.
 * @param b The second `tf.Tensor` to add. Must have the same type as `a`.
 *
 * @doc {heading: 'Operations', subheading: 'Arithmetic'}
 */
declare function add_<T extends Tensor>(a: Tensor | TensorLike, b: Tensor | TensorLike): T;
declare const add: typeof add_;
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/gradients/AddN_grad" />
declare const addNGradConfig: GradConfig;
/// <amd-module name="@tensorflow/tfjs-core/dist/gradients/Add_grad" />
declare const addGradConfig: GradConfig;
/// <amd-module name="@tensorflow/tfjs-core/dist/ops/add_n" />
/**
 * Adds a list of `tf.Tensor`s element-wise, each with the same shape and dtype.
 *
 * ```js
 * const a = tf.tensor1d([1, 2]);
 * const b = tf.tensor1d([3, 4]);
 * const c = tf.tensor1d([5, 6]);
 *
 * tf.addN([a, b, c]).print();
 * ```
 * @param tensors A list of tensors with the same shape and dtype.
 * @doc {heading: 'Operations', subheading: 'Arithmetic'}
 */
declare function addN_<T extends Tensor>(tensors: Array<T | TensorLike>): T;
declare const addN: typeof addN_;
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/add_n_test" />
{ };
/// <amd-module name="@tensorflow/tfjs-core/dist/ops/add_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-layers/dist/layers/advanced_activations" />
/**
 *  Advanced activation layers.
 */
declare interface ReLULayerArgs extends LayerArgs {
    /**
     * Float, the maximum output value.
     */
    maxValue?: number;
}
declare class ReLU extends Layer {
    /** @nocollapse */
    static className: string;
    maxValue: number;
    constructor(args?: ReLULayerArgs);
    call(inputs: Tensor | Tensor[], kwargs: Kwargs): Tensor | Tensor[];
    computeOutputShape(inputShape: Shape | Shape[]): Shape | Shape[];
    getConfig(): serialization.ConfigDict;
}
declare interface LeakyReLULayerArgs extends LayerArgs {
    /**
     * Float `>= 0`. Negative slope coefficient. Defaults to `0.3`.
     */
    alpha?: number;
}
declare class LeakyReLU extends Layer {
    /** @nocollapse */
    static className: string;
    readonly alpha: number;
    readonly DEFAULT_ALPHA = 0.3;
    constructor(args?: LeakyReLULayerArgs);
    call(inputs: Tensor | Tensor[], kwargs: Kwargs): Tensor | Tensor[];
    computeOutputShape(inputShape: Shape | Shape[]): Shape | Shape[];
    getConfig(): serialization.ConfigDict;
}
declare interface PReLULayerArgs extends LayerArgs {
    /**
     * Initializer for the learnable alpha.
     */
    alphaInitializer?: Initializer | InitializerIdentifier;
    /**
     * Regularizer for the learnable alpha.
     */
    alphaRegularizer?: Regularizer;
    /**
     * Constraint for the learnable alpha.
     */
    alphaConstraint?: Constraint;
    /**
     * The axes along which to share learnable parameters for the activation
     * function. For example, if the incoming feature maps are from a 2D
     * convolution with output shape `[numExamples, height, width, channels]`,
     * and you wish to share parameters across space (height and width) so that
     * each filter channels has only one set of parameters, set
     * `shared_axes: [1, 2]`.
     */
    sharedAxes?: number | number[];
}
declare class PReLU extends Layer {
    /** @nocollapse */
    static className: string;
    private readonly alphaInitializer;
    private readonly alphaRegularizer;
    private readonly alphaConstraint;
    private readonly sharedAxes;
    private alpha;
    readonly DEFAULT_ALPHA_INITIALIZER: InitializerIdentifier;
    constructor(args?: PReLULayerArgs);
    build(inputShape: Shape | Shape[]): void;
    call(inputs: Tensor | Tensor[], kwargs: Kwargs): Tensor | Tensor[];
    getConfig(): serialization.ConfigDict;
}
declare interface ELULayerArgs extends LayerArgs {
    /**
     * Float `>= 0`. Negative slope coefficient. Defaults to `1.0`.
     */
    alpha?: number;
}
declare class ELU extends Layer {
    /** @nocollapse */
    static className: string;
    readonly alpha: number;
    readonly DEFAULT_ALPHA = 1;
    constructor(args?: ELULayerArgs);
    call(inputs: Tensor | Tensor[], kwargs: Kwargs): Tensor | Tensor[];
    computeOutputShape(inputShape: Shape | Shape[]): Shape | Shape[];
    getConfig(): serialization.ConfigDict;
}
declare interface ThresholdedReLULayerArgs extends LayerArgs {
    /**
     * Float >= 0. Threshold location of activation.
     */
    theta?: number;
}
declare class ThresholdedReLU extends Layer {
    /** @nocollapse */
    static className: string;
    readonly theta: number;
    readonly DEFAULT_THETA = 1;
    constructor(args?: ThresholdedReLULayerArgs);
    call(inputs: Tensor | Tensor[], kwargs: Kwargs): Tensor | Tensor[];
    computeOutputShape(inputShape: Shape | Shape[]): Shape | Shape[];
    getConfig(): serialization.ConfigDict;
}
declare interface SoftmaxLayerArgs extends LayerArgs {
    /**
     * Integer, axis along which the softmax normalization is applied.
     * Defaults to `-1` (i.e., the last axis).
     */
    axis?: number;
}
declare class Softmax extends Layer {
    /** @nocollapse */
    static className: string;
    readonly axis: number;
    readonly softmax: (t: Tensor, a?: number) => Tensor;
    readonly DEFAULT_AXIS = 1;
    constructor(args?: SoftmaxLayerArgs);
    call(inputs: Tensor | Tensor[], kwargs: Kwargs): Tensor | Tensor[];
    computeOutputShape(inputShape: Shape | Shape[]): Shape | Shape[];
    getConfig(): serialization.ConfigDict;
}

/// <amd-module name="@tensorflow/tfjs-layers/dist/keras_format/layers/advanced_activation_serialization" />
interface ReLULayerConfig extends LayerConfig {
    max_value?: number;
}
declare type ReLULayerSerialization = BaseLayerSerialization<'ReLU', ReLULayerConfig>;
interface LeakyReLULayerConfig extends LayerConfig {
    alpha?: number;
}
declare type LeakyReLULayerSerialization = BaseLayerSerialization<'LeakyReLU', LeakyReLULayerConfig>;
interface PReLULayerConfig extends LayerConfig {
    alpha_initializer?: InitializerSerialization;
    alpha_regularizer?: RegularizerSerialization;
    alpha_constraint?: ConstraintSerialization;
    shared_axes?: number | number[];
}
declare type PReLULayerSerialization = BaseLayerSerialization<'PReLU', PReLULayerConfig>;
interface ELULayerConfig extends LayerConfig {
    alpha?: number;
}
declare type ELULayerSerialization = BaseLayerSerialization<'ELU', ELULayerConfig>;
interface ThresholdedReLULayerConfig extends LayerConfig {
    theta?: number;
}
declare type ThresholdedReLULayerSerialization = BaseLayerSerialization<'ThresholdedReLU', ThresholdedReLULayerConfig>;
interface SoftmaxLayerConfig extends LayerConfig {
    axis?: number;
}
declare type SoftmaxLayerSerialization = BaseLayerSerialization<'Softmax', SoftmaxLayerConfig>;
declare type AdvancedActivationLayerSerialization = ReLULayerSerialization | LeakyReLULayerSerialization | PReLULayerSerialization | ELULayerSerialization | ThresholdedReLULayerSerialization | SoftmaxLayerSerialization;
declare type AdvancedActivationLayerClassName = AdvancedActivationLayerSerialization['class_name'];
/**
 * A string array of valid AdvancedActivationLayer class names.
 *
 * This is guaranteed to match the `AdvancedActivationLayerClassName` union
 * type.
 */
declare const advancedActivationLayerClassNames: AdvancedActivationLayerClassName[];

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/all" />
/**
 * Computes the logical and of elements across dimensions of a `tf.Tensor`.
 *
 * Reduces the input along the dimensions given in `axes`. Unless `keepDims`
 * is true, the rank of the `tf.Tensor` is reduced by 1 for each entry in
 * `axes`. If `keepDims` is true, the reduced dimensions are retained with
 * length 1. If `axes` has no entries, all dimensions are reduced, and a
 * `tf.Tensor` with a single element is returned.
 *
 * ```js
 * const x = tf.tensor1d([1, 1, 1], 'bool');
 *
 * x.all().print();  // or tf.all(x)
 * ```
 *
 * ```js
 * const x = tf.tensor2d([1, 1, 0, 0], [2, 2], 'bool');
 *
 * const axis = 1;
 * x.all(axis).print();  // or tf.all(x, axis)
 * ```
 *
 * @param x The input tensor. Must be of dtype bool.
 * @param axis The dimension(s) to reduce. By default it reduces
 *     all dimensions.
 * @param keepDims If true, retains reduced dimensions with size 1.
 *
 * @doc {heading: 'Operations', subheading: 'Reduction'}
 */
declare function all_<T extends Tensor>(x: Tensor | TensorLike, axis?: number | number[], keepDims?: boolean): T;
declare const all: typeof all_;
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/all_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/any" />
/**
 * Computes the logical or of elements across dimensions of a `tf.Tensor`.
 *
 * Reduces the input along the dimensions given in `axes`. Unless `keepDims`
 * is true, the rank of the `tf.Tensor` is reduced by 1 for each entry in
 * `axes`. If `keepDims` is true, the reduced dimensions are retained with
 * length 1. If `axes` has no entries, all dimensions are reduced, and a
 * `tf.Tensor` with a single element is returned.
 *
 * ```js
 * const x = tf.tensor1d([1, 1, 1], 'bool');
 *
 * x.any().print();  // or tf.any(x)
 * ```
 *
 * ```js
 * const x = tf.tensor2d([1, 1, 0, 0], [2, 2], 'bool');
 *
 * const axis = 1;
 * x.any(axis).print();  // or tf.any(x, axis)
 * ```
 *
 * @param x The input tensor. Must be of dtype bool.
 * @param axis The dimension(s) to reduce. By default it reduces
 *     all dimensions.
 * @param keepDims If true, retains reduced dimensions with size 1.
 *
 * @doc {heading: 'Operations', subheading: 'Reduction'}
 */
declare function any_<T extends Tensor>(x: Tensor | TensorLike, axis?: number | number[], keepDims?: boolean): T;
declare const any: typeof any_;
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/any_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/gradients/ArgMax_grad" />
declare const argMaxGradConfig: GradConfig;

/// <amd-module name="@tensorflow/tfjs-core/dist/gradients/ArgMin_grad" />
declare const argMinGradConfig: GradConfig;

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/arg_max" />
/**
 * Returns the indices of the maximum values along an `axis`.
 *
 * The result has the same shape as `input` with the dimension along `axis`
 * removed.
 *
 * ```js
 * const x = tf.tensor1d([1, 2, 3]);
 *
 * x.argMax().print();  // or tf.argMax(x)
 * ```
 *
 * ```js
 * const x = tf.tensor2d([1, 2, 4, 3], [2, 2]);
 *
 * const axis = 1;
 * x.argMax(axis).print();  // or tf.argMax(x, axis)
 * ```
 *
 * @param x The input tensor.
 * @param axis The dimension to reduce. Defaults to 0 (outer-most dimension).
 *
 * @doc {heading: 'Operations', subheading: 'Reduction'}
 */
declare function argMax_<T extends Tensor>(x: Tensor | TensorLike, axis?: number): T;
declare const argMax: typeof argMax_;
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/arg_max_test" />
{ };
/// <amd-module name="@tensorflow/tfjs-core/dist/public/chained_ops/arg_min" />
declare module '../../tensor' {
    interface Tensor<R extends Rank = Rank> {
        argMin<T extends Tensor>(axis?: number): T;
    }
}

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/arg_min_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/arithmetic_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/array_ops_util" />
/**
 * Gets the new shape of the input Tensor after it's been reshaped
 * to:
 * [blockShape[0], ..., blockShape[M-1], batch / prod(blockShape),
 * inputShape[1], ..., inputShape[N-1]]
 *
 * See step 1: https://www.tensorflow.org/api_docs/python/tf/batch_to_space_nd
 */
declare function getReshaped(inputShape: number[], blockShape: number[], prod: number, batchToSpace?: boolean): number[];
/**
 * Gets the permutation that will transpose the dimensions of the
 * reshaped tensor to shape:
 *
 * [batch / prod(block_shape),inputShape[1], blockShape[0], ...,
 * inputShape[M], blockShape[M-1],inputShape[M+1], ..., inputShape[N-1]]
 *
 * see step 2: https://www.tensorflow.org/api_docs/python/tf/batch_to_space_nd
 */
declare function getPermuted(reshapedRank: number, blockShapeRank: number, batchToSpace?: boolean): number[];
/**
 * Gets the shape of the reshaped and permuted input Tensor before any cropping
 * is applied.  The new shape will be:
 *
 * [batch / prod(blockShape),inputShape[1] * blockShape[0], ...,
 * inputShape[M] * blockShape[M-1],inputShape[M+1], ..., inputShape[N-1]]
 *
 * See step 3: https://www.tensorflow.org/api_docs/python/tf/batch_to_space_nd
 */
declare function getReshapedPermuted(inputShape: number[], blockShape: number[], prod: number, batchToSpace?: boolean): number[];
/**
 * Converts the crops argument into the beginning coordinates of a slice
 * operation.
 */
declare function getSliceBeginCoords(crops: number[][], blockShape: number): number[];
/**
 * Converts the crops argument into the size of a slice operation.  When
 * combined with getSliceBeginCoords this function allows the reshaped and
 * permuted Tensor to be cropped to its final output shape of:
 *
 * inputShape[1] * blockShape[0] - crops[0,0] - crops[0,1], ...,
 * inputShape[M] * blockShape[M-1] -crops[M-1,0] -
 * crops[M-1,1],inputShape[M+1], ..., inputShape[N-1]]
 *
 * See step 4: https://www.tensorflow.org/api_docs/python/tf/batch_to_space_nd
 */
declare function getSliceSize(uncroppedShape: number[], crops: number[][], blockShape: number): number[];

/// <amd-module name="@tensorflow/tfjs-core/dist/public/chained_ops/as1d" />
declare module '../../tensor' {
    interface Tensor<R extends Rank = Rank> {
        as1D<T extends Tensor>(): Tensor1D;
    }
}

/// <amd-module name="@tensorflow/tfjs-core/dist/public/chained_ops/as2d" />
declare module '../../tensor' {
    interface Tensor<R extends Rank = Rank> {
        as2D<T extends Tensor>(rows: number, columns: number): Tensor2D;
    }
}

/// <amd-module name="@tensorflow/tfjs-core/dist/public/chained_ops/as3d" />
declare module '../../tensor' {
    interface Tensor<R extends Rank = Rank> {
        as3D<T extends Tensor>(rows: number, columns: number, depth: number): Tensor3D;
    }
}

/// <amd-module name="@tensorflow/tfjs-core/dist/public/chained_ops/as4d" />
declare module '../../tensor' {
    interface Tensor<R extends Rank = Rank> {
        as4D<T extends Tensor>(rows: number, columns: number, depth: number, depth2: number): Tensor4D;
    }
}

/// <amd-module name="@tensorflow/tfjs-core/dist/public/chained_ops/as5d" />
declare module '../../tensor' {
    interface Tensor<R extends Rank = Rank> {
        as5D<T extends Tensor>(rows: number, columns: number, depth: number, depth2: number, depth3: number): Tensor5D;
    }
}

/// <amd-module name="@tensorflow/tfjs-core/dist/public/chained_ops/asin" />
declare module '../../tensor' {
    interface Tensor<R extends Rank = Rank> {
        asin<T extends Tensor>(this: T): T;
    }
}

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/asinh" />
/**
 * Computes inverse hyperbolic sin of the input `tf.Tensor` element-wise:
 * `asinh(x)`
 *
 * ```js
 * const x = tf.tensor1d([0, 1, -1, .7]);
 *
 * x.asinh().print();  // or tf.asinh(x)
 * ```
 * @param x The input tensor.
 *
 * @doc {heading: 'Operations', subheading: 'Basic math'}
 */
declare function asinh_<T extends Tensor>(x: T | TensorLike): T;
declare const asinh: typeof asinh_;
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/gradients/Asinh_grad" />
declare const asinhGradConfig: GradConfig;

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/asinh_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/gradients/Asin_grad" />
declare const asinGradConfig: GradConfig;

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/asin_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/public/chained_ops/as_scalar" />
declare module '../../tensor' {
    interface Tensor<R extends Rank = Rank> {
        asScalar<T extends Tensor>(): Scalar;
    }
}

/// <amd-module name="@tensorflow/tfjs-core/dist/public/chained_ops/as_type" />
declare module '../../tensor' {
    interface Tensor<R extends Rank = Rank> {
        asType<T extends Tensor>(this: T, dtype: DataType): T;
    }
}

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/atan" />
/**
 * Computes atan of the input `tf.Tensor` element-wise: `atan(x)`
 *
 * ```js
 * const x = tf.tensor1d([0, 1, -1, .7]);
 *
 * x.atan().print();  // or tf.atan(x)
 * ```
 * @param x The input tensor.
 *
 * @doc {heading: 'Operations', subheading: 'Basic math'}
 */
declare function atan_<T extends Tensor>(x: T | TensorLike): T;
declare const atan: typeof atan_;
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/atan2" />
/**
 * Computes arctangent of `tf.Tensor`s a / b element-wise: `atan2(a, b)`.
 * Supports broadcasting.
 *
 * ```js
 * const a = tf.tensor1d([1.0, 1.0, -1.0, .7]);
 * const b = tf.tensor1d([2.0, 13.0, 3.5, .21]);
 *
 * tf.atan2(a, b).print()
 * ```
 *
 * @param a The first tensor.
 * @param b The second tensor. Must have the same dtype as `a`.
 *
 * @doc {heading: 'Operations', subheading: 'Basic math'}
 */
declare function atan2_<T extends Tensor>(a: Tensor | TensorLike, b: Tensor | TensorLike): T;
declare const atan2: typeof atan2_;
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/gradients/Atan2_grad" />
declare const atan2GradConfig: GradConfig;

/// <amd-module name="@tensorflow/tfjs-core/dist/public/chained_ops/atanh" />
declare module '../../tensor' {
    interface Tensor<R extends Rank = Rank> {
        atanh<T extends Tensor>(this: T): T;
    }
}

/// <amd-module name="@tensorflow/tfjs-core/dist/gradients/Atanh_grad" />
declare const atanhGradConfig: GradConfig;

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/atanh_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/gradients/Atan_grad" />
declare const atanGradConfig: GradConfig;

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/atan_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/gradients/AvgPool3D_grad" />
declare const avgPool3DGradConfig: GradConfig;

/// <amd-module name="@tensorflow/tfjs-core/dist/gradients/AvgPool_grad" />
declare const avgPoolGradConfig: GradConfig;
/// <amd-module name="@tensorflow/tfjs-core/dist/public/chained_ops/avg_pool" />

declare module '../../tensor' {
    interface Tensor<R extends Rank = Rank> {
        avgPool<T extends Tensor3D | Tensor4D>(filterSize: [number, number] | number, strides: [number, number] | number, pad: 'valid' | 'same' | number | ExplicitPadding, dimRoundingMode?: 'floor' | 'round' | 'ceil'): T;
    }
}

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/avg_pool_3d" />
/**
 * Computes the 3D average pooling.
 *
 * ```js
 * const x = tf.tensor5d([1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 2, 2, 1]);
 * const result = tf.avgPool3d(x, 2, 1, 'valid');
 * result.print();
 * ```
 *
 * @param x The input tensor, of rank 5 or rank 4 of shape
 *     `[batch, depth, height, width, inChannels]`.
 * @param filterSize The filter size:
 *     `[filterDepth, filterHeight, filterWidth]`.
 *     If `filterSize` is a single number,
 *     then `filterDepth == filterHeight == filterWidth`.
 * @param strides The strides of the pooling:
 *     `[strideDepth, strideHeight, strideWidth]`.
 *     If `strides` is a single number,
 *     then `strideDepth == strideHeight == strideWidth`.
 * @param pad The type of padding algorithm.
 *    - `same` and stride 1: output will be of same size as input,
 *       regardless of filter size.
 *    - `valid`: output will be smaller than input if filter is larger
 *       than 1*1x1.
 *    - For more info, see this guide:
 *     [https://www.tensorflow.org/api_docs/python/tf/nn/convolution](
 *          https://www.tensorflow.org/api_docs/python/tf/nn/convolution)
 * @param dimRoundingMode A string from: 'ceil', 'round', 'floor'. If none is
 *     provided, it will default to truncate.
 * @param dataFormat An optional string from: "NDHWC", "NCDHW". Defaults to
 *     "NDHWC". Specify the data format of the input and output data. With the
 *     default format "NDHWC", the data is stored in the order of: [batch,
 *     depth, height, width, channels]. Only "NDHWC" is currently supported.
 *
 * @doc {heading: 'Operations', subheading: 'Convolution'}
 */
declare function avgPool3d_<T extends Tensor4D | Tensor5D>(x: T | TensorLike, filterSize: [number, number, number] | number, strides: [number, number, number] | number, pad: 'valid' | 'same' | number, dimRoundingMode?: 'floor' | 'round' | 'ceil', dataFormat?: 'NDHWC' | 'NCDHW'): T;
declare const avgPool3d: typeof avgPool3d_;
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/avg_pool_3d_grad" />
/**
 * Computes the backprop of a 3d avg pool.
 *
 * @param dy The dy error, of rank 5 of shape
 *     [batchSize, depth, height, width, channels].
 * assumed.
 * @param input The original input image, of rank 5 or rank4 of shape
 *     [batchSize, depth, height, width, channels].
 * @param filterSize The filter size:
 *     `[filterDepth, filterHeight, filterWidth]`.
 *     `filterSize` is a single number,
 *     then `filterDepth == filterHeight == filterWidth`.
 * @param strides The strides of the pooling:
 *     `[strideDepth, strideHeight, strideWidth]`. If
 *     `strides` is a single number, then `strideHeight == strideWidth`.
 * @param pad A string from: 'same', 'valid'. The type of padding algorithm
 *     used in the forward prop of the op.
 * @param dimRoundingMode A string from: 'ceil', 'round', 'floor'. If none is
 *     provided, it will default to truncate.
 */
declare function avgPool3dGrad_<T extends Tensor4D | Tensor5D>(dy: T | TensorLike, input: T | TensorLike, filterSize: [number, number, number] | number, strides: [number, number, number] | number, pad: 'valid' | 'same' | number, dimRoundingMode?: 'floor' | 'round' | 'ceil'): T;
declare const avgPool3dGrad: typeof avgPool3dGrad_;
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/avg_pool_3d_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/avg_pool_grad" />
/**
 * Computes the backprop of an 2D avg pool.
 *
 * @param dy The dy error, of rank 4 or rank 3 of shape
 *     [batchSize, height, width, channels]. If rank 3, batch of 1 is
 * assumed.
 * @param input The input image, of rank 4 or rank 3 of shape
 *     [batchSize, height, width, channels]. If rank 3, batch of 1 is
 * assumed.
 * @param filterSize The filter size: `[filterHeight, filterWidth]`. If
 *     `filterSize` is a single number, then `filterHeight == filterWidth`.
 * @param strides The strides of the pooling: `[strideHeight, strideWidth]`. If
 *     `strides` is a single number, then `strideHeight == strideWidth`.
 * @param pad The type of padding algorithm used in the forward prop of the op.
 *     'same', 'valid', for more info, see this guide:
 *     [https://www.tensorflow.org/api_docs/python/tf/nn/convolution](
 *         https://www.tensorflow.org/api_docs/python/tf/nn/convolution)
 */
declare function avgPoolGrad_<T extends Tensor3D | Tensor4D>(dy: T | TensorLike, input: T | TensorLike, filterSize: [number, number] | number, strides: [number, number] | number, pad: 'valid' | 'same' | number | ExplicitPadding): T;
declare const avgPoolGrad: typeof avgPoolGrad_;
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/avg_pool_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/axis_util" />
/**
 * Returns true if the axis specifies the inner most dimensions of the
 * array.
 */
declare function axesAreInnerMostDims(axes: number[], rank: number): boolean;
declare function combineLocations(outputLoc: number[], reduceLoc: number[], axes: number[]): number[];
declare function computeOutAndReduceShapes(aShape: number[], axes: number[]): [number[], number[]];
declare function expandShapeToKeepDim(shape: number[], axes: number[]): number[];
declare function assertAxesAreInnerMostDims(msg: string, axes: number[], rank: number): void;
/**
 * Returns the axes permutation to be used with `tf.transpose`, if such
 * permutation is necessary. Otherwise it returns null. This method is used by
 * operations that operate only on inner-most axes.
 */
declare function getAxesPermutation(axes: number[], rank: number): number[] | null;
/** Returns the axes permutation that undoes the original permutation. */
declare function getUndoAxesPermutation(axes: number[]): number[];
declare function getInnerMostAxes(numAxes: number, rank: number): number[];

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/axis_util_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/backends/backend" />
declare const EPSILON_FLOAT32 = 1e-7;
declare const EPSILON_FLOAT16 = 0.0001;
interface BackendTimingInfo {
    kernelMs: number | {
        error: string;
    };
    getExtraProfileInfo?(): string;
}
interface TensorStorage {
    read(dataId: DataId): Promise<BackendValues>;
    readSync(dataId: DataId): BackendValues;
    disposeData(dataId: DataId, force?: boolean): boolean;
    write(values: BackendValues, shape: number[], dtype: DataType): DataId;
    move(dataId: DataId, values: BackendValues, shape: number[], dtype: DataType, refCount: number): void;
    memory(): {
        unreliable: boolean;
    };
    /** Returns number of data ids currently in the storage. */
    numDataIds(): number;
    refCount(dataId: DataId): number;
}
/** Convenient class for storing tensor-related data. */
declare class DataStorage<T> {
    private backend;
    private dataMover;
    private data;
    private dataIdsCount;
    constructor(backend: KernelBackend, dataMover: DataMover);
    get(dataId: DataId): T;
    set(dataId: DataId, value: T): void;
    has(dataId: DataId): boolean;
    delete(dataId: DataId): boolean;
    numDataIds(): number;
}
interface DataMover {
    /**
     * To be called by backends whenever they see a dataId that they don't own.
     * Upon calling this method, the mover will fetch the tensor from another
     * backend and register it with the current active backend.
     */
    moveData(backend: KernelBackend, dataId: DataId): void;
}
interface BackendTimer {
    timerAvailable(): boolean;
    time(f: () => void): Promise<BackendTimingInfo>;
}
/**
 * The interface that defines the kernels that should be implemented when
 * adding a new backend. New backends don't need to implement every one of the
 * methods, this can be done gradually (throw an error for unimplemented
 * methods).
 */
declare class KernelBackend implements TensorStorage, Backend, BackendTimer {
    refCount(dataId: DataId): number;
    incRef(dataId: DataId): void;
    timerAvailable(): boolean;
    time(f: () => void): Promise<BackendTimingInfo>;
    read(dataId: object): Promise<BackendValues>;
    readSync(dataId: object): BackendValues;
    readToGPU(dataId: object, options?: DataToGPUOptions): GPUData;
    numDataIds(): number;
    disposeData(dataId: object, force?: boolean): boolean;
    write(values: BackendValues, shape: number[], dtype: DataType): DataId;
    move(dataId: DataId, values: BackendValues, shape: number[], dtype: DataType, refCount: number): void;
    createTensorFromGPUData(values: WebGLData | WebGPUData, shape: number[], dtype: DataType): Tensor;
    memory(): {
        unreliable: boolean;
        reasons?: string[];
    };
    /** Returns the highest precision for floats in bits (e.g. 16 or 32) */
    floatPrecision(): 16 | 32;
    /** Returns the smallest representable number.  */
    epsilon(): number;
    dispose(): void;
}
/// <amd-module name="@tensorflow/tfjs-core/dist/backends/backend_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/backends/backend_util" />

{ segment_util };
declare function fromUint8ToStringArray(vals: Uint8Array[]): string[];
declare function fromStringArrayToUint8(strings: string[]): Uint8Array[];

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/linalg/band_part" />
/**
 * Copy a tensor setting everything outside a central band in each innermost
 * matrix to zero.
 *
 * The band part is computed as follows: Assume input has `k` dimensions
 * `[I, J, K, ..., M, N]`, then the output is a tensor with the same shape where
 * `band[i, j, k, ..., m, n] = in_band(m, n) * input[i, j, k, ..., m, n]`.
 * The indicator function
 * `in_band(m, n) = (num_lower < 0 || (m-n) <= num_lower)`
 * `&& (num_upper < 0 || (n-m) <= num_upper)`
 *
 * ```js
 * const x = tf.tensor2d([[ 0,  1,  2, 3],
 *                        [-1,  0,  1, 2],
 *                        [-2, -1,  0, 1],
 *                        [-3, -2, -1, 0]]);
 * let y = tf.linalg.bandPart(x, 1, -1);
 * y.print(); // [[ 0,  1,  2, 3],
 *            //  [-1,  0,  1, 2],
 *            //  [ 0, -1,  0, 1],
 *            //  [ 0, 0 , -1, 0]]
 * let z = tf.linalg.bandPart(x, 2, 1);
 * z.print(); // [[ 0,  1,  0, 0],
 *            //  [-1,  0,  1, 0],
 *            //  [-2, -1,  0, 1],
 *            //  [ 0, -2, -1, 0]]
 * ```
 *
 * @param x Rank `k` tensor
 * @param numLower Number of subdiagonals to keep.
 *   If negative, keep entire lower triangle.
 * @param numUpper Number of subdiagonals to keep.
 *   If negative, keep entire upper triangle.
 * @returns Rank `k` tensor of the same shape as input.
 *   The extracted banded tensor.
 *
 * @doc {heading:'Operations', subheading:'Linear Algebra', namespace:'linalg'}
 */
declare function bandPart_<T extends Tensor>(a: T | TensorLike, numLower: number, numUpper: number): T;
declare const bandPart: typeof bandPart_;
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/linalg/band_part_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/base" />
/**
 * @fileoverview
 * @suppress {partialAlias} Optimization disabled due to passing the module
 * object into a function below:
 *
 *   import * as ops from './ops/ops';
 *   setOpHandler(ops);
 */


/// <amd-module name="@tensorflow/tfjs-layers/dist/base_callbacks" />
/** Verbosity logging level when fitting a model. */
declare enum ModelLoggingVerbosity {
    SILENT = 0,
    VERBOSE = 1
}
/** How often to yield to the main thread when training (in ms). */
declare const DEFAULT_YIELD_EVERY_MS = 125;
declare type Params = {
    [key: string]: number | string | boolean | number[] | string[] | boolean[];
};
declare type YieldEveryOptions = 'auto' | 'batch' | 'epoch' | 'never' | number;
/**
 * Abstract base class used to build new callbacks.
 *
 * The `logs` dictionary that callback methods take as argument will contain
 * keys for quantities relevant to the current batch or epoch.
 *
 * Currently, the `.fit()` method of the `Sequential` model class
 * will include the following quantities in the `logs` that
 * it passes to its callbacks:
 *
 * onEpochEnd: Logs include `acc` and `loss`, and optionally include `valLoss`
 *   (if validation is enabled in `fit`), and `valAcc` (if validation and
 *   accuracy monitoring are enabled).
 * onBatchBegin: Logs include `size`, the number of samples in the current
 *   batch.
 * onBatchEnd: Logs include `loss`, and optionally `acc` (if accuracy monitoring
 *   is enabled).
 */
declare abstract class BaseCallback {
    validationData: Tensor | Tensor[];
    /**
     * Training parameters (eg. verbosity, batch size, number of epochs...).
     */
    params: Params;
    setParams(params: Params): void;
    onEpochBegin(epoch: number, logs?: UnresolvedLogs): Promise<void>;
    onEpochEnd(epoch: number, logs?: UnresolvedLogs): Promise<void>;
    onBatchBegin(batch: number, logs?: UnresolvedLogs): Promise<void>;
    onBatchEnd(batch: number, logs?: UnresolvedLogs): Promise<void>;
    onTrainBegin(logs?: UnresolvedLogs): Promise<void>;
    onTrainEnd(logs?: UnresolvedLogs): Promise<void>;
    setModel(model: Container): void;
}
/**
 * Container abstracting a list of callbacks.
 */
declare class CallbackList {
    callbacks: BaseCallback[];
    queueLength: number;
    /**
     * Constructor of CallbackList.
     * @param callbacks Array of `Callback` instances.
     * @param queueLength Queue length for keeping running statistics over
     *   callback execution time.
     */
    constructor(callbacks?: BaseCallback[], queueLength?: number);
    append(callback: BaseCallback): void;
    setParams(params: Params): void;
    setModel(model: Container): void;
    /**
     * Called at the start of an epoch.
     * @param epoch Index of epoch.
     * @param logs Dictionary of logs.
     */
    onEpochBegin(epoch: number, logs?: UnresolvedLogs): Promise<void>;
    /**
     * Called at the end of an epoch.
     * @param epoch Index of epoch.
     * @param logs Dictionary of logs.
     */
    onEpochEnd(epoch: number, logs?: UnresolvedLogs): Promise<void>;
    /**
     * Called  right before processing a batch.
     * @param batch Index of batch within the current epoch.
     * @param logs Dictionary of logs.
     */
    onBatchBegin(batch: number, logs?: UnresolvedLogs): Promise<void>;
    /**
     * Called at the end of a batch.
     * @param batch Index of batch within the current epoch.
     * @param logs Dictionary of logs.
     */
    onBatchEnd(batch: number, logs?: UnresolvedLogs): Promise<void>;
    /**
     * Called at the beginning of training.
     * @param logs Dictionary of logs.
     */
    onTrainBegin(logs?: UnresolvedLogs): Promise<void>;
    /**
     * Called at the end of training.
     * @param logs Dictionary of logs.
     */
    onTrainEnd(logs?: UnresolvedLogs): Promise<void>;
}
/**
 * Callback that accumulates epoch averages of metrics.
 *
 * This callback is automatically applied to every LayersModel.
 */
declare class BaseLogger extends BaseCallback {
    private seen;
    private totals;
    constructor();
    onEpochBegin(epoch: number): Promise<void>;
    onBatchEnd(batch: number, logs?: UnresolvedLogs): Promise<void>;
    onEpochEnd(epoch: number, logs?: UnresolvedLogs): Promise<void>;
}
/**
 * Callback that records events into a `History` object. This callback is
 * automatically applied to every TF.js Layers model. The `History` object
 * gets returned by the `fit` method of models.
 */
declare class History extends BaseCallback {
    epoch: number[];
    history: {
        [key: string]: Array<number | Tensor>;
    };
    onTrainBegin(logs?: UnresolvedLogs): Promise<void>;
    onEpochEnd(epoch: number, logs?: UnresolvedLogs): Promise<void>;
    /**
     * Await the values of all losses and metrics.
     */
    syncData(): Promise<void>;
}
interface CustomCallbackArgs {
    onTrainBegin?: (logs?: Logs) => void | Promise<void>;
    onTrainEnd?: (logs?: Logs) => void | Promise<void>;
    onEpochBegin?: (epoch: number, logs?: Logs) => void | Promise<void>;
    onEpochEnd?: (epoch: number, logs?: Logs) => void | Promise<void>;
    onBatchBegin?: (batch: number, logs?: Logs) => void | Promise<void>;
    onBatchEnd?: (batch: number, logs?: Logs) => void | Promise<void>;
    onYield?: (epoch: number, batch: number, logs: Logs) => void | Promise<void>;
    nowFunc?: Function;
    nextFrameFunc?: Function;
}
/**
 * Custom callback for training.
 */
declare class CustomCallback extends BaseCallback {
    protected readonly trainBegin: (logs?: Logs) => void | Promise<void>;
    protected readonly trainEnd: (logs?: Logs) => void | Promise<void>;
    protected readonly epochBegin: (epoch: number, logs?: Logs) => void | Promise<void>;
    protected readonly epochEnd: (epoch: number, logs?: Logs) => void | Promise<void>;
    protected readonly batchBegin: (batch: number, logs?: Logs) => void | Promise<void>;
    protected readonly batchEnd: (batch: number, logs?: Logs) => void | Promise<void>;
    protected readonly yield: (epoch: number, batch: number, logs: Logs) => void | Promise<void>;
    private yieldEvery;
    private currentEpoch;
    nowFunc: Function;
    nextFrameFunc: Function;
    constructor(args: CustomCallbackArgs, yieldEvery?: YieldEveryOptions);
    maybeWait(epoch: number, batch: number, logs: UnresolvedLogs): Promise<void>;
    onEpochBegin(epoch: number, logs?: UnresolvedLogs): Promise<void>;
    onEpochEnd(epoch: number, logs?: UnresolvedLogs): Promise<void>;
    onBatchBegin(batch: number, logs?: UnresolvedLogs): Promise<void>;
    onBatchEnd(batch: number, logs?: UnresolvedLogs): Promise<void>;
    onTrainBegin(logs?: UnresolvedLogs): Promise<void>;
    onTrainEnd(logs?: UnresolvedLogs): Promise<void>;
}
/**
 * Standardize callbacks or configurations of them to an Array of callbacks.
 */
declare function standardizeCallbacks(callbacks: BaseCallback | BaseCallback[] | CustomCallbackArgs | CustomCallbackArgs[], yieldEvery: YieldEveryOptions): BaseCallback[];
declare type BaseCallbackConstructor = {
    new(): BaseCallback;
};
/**
 * A global registry for callback constructors to be used during
 * LayersModel.fit().
 */
declare class CallbackConstructorRegistry {
    private static constructors;
    /**
     * Blocks public access to constructor.
     */
    private constructor();
    /**
     * Register a tf.LayersModel.fit() callback constructor.
     *
     * The registered callback constructor will be used to instantiate
     * callbacks for every tf.LayersModel.fit() call afterwards.
     *
     * @param verbosityLevel Level of verbosity at which the `callbackConstructor`
     *   is to be reigstered.
     * @param callbackConstructor A no-arg constructor for `tf.Callback`.
     * @throws Error, if the same callbackConstructor has been registered before,
     *   either at the same or a different `verbosityLevel`.
     */
    static registerCallbackConstructor(verbosityLevel: number, callbackConstructor: BaseCallbackConstructor): void;
    private static checkForDuplicate;
    /**
     * Clear all registered callback constructors.
     */
    protected static clear(): void;
    /**
     * Create callbacks using the registered callback constructors.
     *
     * Given `verbosityLevel`, all constructors registered at that level or above
     * will be called and the instantiated callbacks will be used.
     *
     * @param verbosityLevel: Level of verbosity.
     */
    static createCallbacks(verbosityLevel: number): BaseCallback[];
}
declare function configureCallbacks(callbacks: BaseCallback[], verbose: ModelLoggingVerbosity, epochs: number, initialEpoch: number, numTrainSamples: number, stepsPerEpoch: number, batchSize: number, doValidation: boolean, callbackMetrics: string[]): {
    callbackList: CallbackList;
    history: History;
};

/// <amd-module name="@tensorflow/tfjs-core/dist/base_side_effects" />

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/basic_lstm_cell" />
/**
 * Computes the next state and output of a BasicLSTMCell.
 *
 * Returns `[newC, newH]`.
 *
 * Derived from tf.contrib.rnn.BasicLSTMCell.
 *
 * @param forgetBias Forget bias for the cell.
 * @param lstmKernel The weights for the cell.
 * @param lstmBias The bias for the cell.
 * @param data The input to the cell.
 * @param c Previous cell state.
 * @param h Previous cell output.
 *
 * @doc {heading: 'Operations', subheading: 'RNN'}
 */
declare function basicLSTMCell_(forgetBias: Scalar | TensorLike, lstmKernel: Tensor2D | TensorLike, lstmBias: Tensor1D | TensorLike, data: Tensor2D | TensorLike, c: Tensor2D | TensorLike, h: Tensor2D | TensorLike): [Tensor2D, Tensor2D];
declare const basicLSTMCell: typeof basicLSTMCell_;
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/basic_lstm_cell_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/gradients/BatchMatMul_grad" />
declare const batchMatMulGradConfig: GradConfig;

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/batchnorm" />
/**
 * Batch normalization.
 *
 * As described in
 * [http://arxiv.org/abs/1502.03167](http://arxiv.org/abs/1502.03167).
 *
 * Mean, variance, scale, and offset can be of two shapes:
 *   - The same shape as the input.
 *   - In the common case, the depth dimension is the last dimension of x, so
 *     the values would be a `tf.Tensor1D` of shape [depth].
 *
 * Also available are stricter rank-specific methods with the same signature
 * as this method that assert that parameters passed are of given rank
 *   - `tf.batchNorm2d`
 *   - `tf.batchNorm3d`
 *   - `tf.batchNorm4d`
 *
 * @param x The input Tensor.
 * @param mean A mean Tensor.
 * @param variance A variance Tensor.
 * @param offset An offset Tensor.
 * @param scale A scale Tensor.
 * @param varianceEpsilon A small float number to avoid dividing by 0.
 *
 * @doc {heading: 'Operations', subheading: 'Normalization'}
 */
declare function batchNorm_<R extends Rank>(x: Tensor<R> | TensorLike, mean: Tensor<R> | Tensor1D | TensorLike, variance: Tensor<R> | Tensor1D | TensorLike, offset?: Tensor<R> | Tensor1D | TensorLike, scale?: Tensor<R> | Tensor1D | TensorLike, varianceEpsilon?: number): Tensor<R>;
declare const batchNorm: typeof batchNorm_;
{ };
/// <amd-module name="@tensorflow/tfjs-core/dist/ops/batchnorm2d" />

/**
 * Batch normalization, strictly for 2D. For the more relaxed version, see
 * `tf.batchNorm`.
 *
 * @param x The input Tensor.
 * @param mean A mean Tensor.
 * @param variance A variance Tensor.
 * @param offset An offset Tensor.
 * @param scale A scale Tensor.
 * @param varianceEpsilon A small float number to avoid dividing by 0.
 */
declare function batchNorm2d_(x: Tensor2D | TensorLike, mean: Tensor2D | Tensor1D | TensorLike, variance: Tensor2D | Tensor1D | TensorLike, offset?: Tensor2D | Tensor1D | TensorLike, scale?: Tensor2D | Tensor1D | TensorLike, varianceEpsilon?: number): Tensor2D;
declare const batchNorm2d: typeof batchNorm2d_;
{ };
/// <amd-module name="@tensorflow/tfjs-core/dist/ops/batchnorm3d" />

/**
 * Batch normalization, strictly for 3D. For the more relaxed version, see
 * `tf.batchNorm`.
 *
 * @param x The input Tensor.
 * @param mean A mean Tensor.
 * @param variance A variance Tensor.
 * @param offset An offset Tensor.
 * @param scale A scale Tensor.
 * @param varianceEpsilon A small float number to avoid dividing by 0.
 */
declare function batchNorm3d_(x: Tensor3D | TensorLike, mean: Tensor3D | Tensor1D | TensorLike, variance: Tensor3D | Tensor1D | TensorLike, offset?: Tensor3D | Tensor1D | TensorLike, scale?: Tensor3D | Tensor1D | TensorLike, varianceEpsilon?: number): Tensor3D;
declare const batchNorm3d: typeof batchNorm3d_;
{ };
/// <amd-module name="@tensorflow/tfjs-core/dist/ops/batchnorm4d" />

/**
 * Batch normalization, strictly for 4D. For the more relaxed version, see
 * `tf.batchNorm`.
 *
 * @param x The input Tensor.
 * @param mean A mean Tensor.
 * @param variance A variance Tensor.
 * @param offset An offset Tensor.
 * @param scale A scale Tensor.
 * @param varianceEpsilon A small float number to avoid dividing by 0.
 */
declare function batchNorm4d_(x: Tensor4D | TensorLike, mean: Tensor4D | Tensor1D | TensorLike, variance: Tensor4D | Tensor1D | TensorLike, offset?: Tensor4D | Tensor1D | TensorLike, scale?: Tensor4D | Tensor1D | TensorLike, varianceEpsilon?: number): Tensor4D;
declare const batchNorm4d: typeof batchNorm4d_;
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/batchnorm_test" />
{ };
/// <amd-module name="@tensorflow/tfjs-core/dist/ops/batchnorm_util" />

declare function xAs4D<R extends Rank>(x: Tensor<R>): Tensor4D;

/// <amd-module name="@tensorflow/tfjs-core/dist/gradients/BatchToSpaceND_grad" />
declare const batchToSpaceNDGradConfig: GradConfig;
/// <amd-module name="@tensorflow/tfjs-core/dist/public/chained_ops/batch_to_space_nd" />
declare module '../../tensor' {
    interface Tensor<R extends Rank = Rank> {
        batchToSpaceND<R extends Rank>(blockShape: number[], crops: number[][]): Tensor<R>;
    }
}

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/batch_to_space_nd_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/binary_ops_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/bincount" />
/**
 * Outputs a vector with length `size` and the same dtype as `weights`.
 *
 * If `weights` are empty, then index `i` stores the number of times the value
 * `i` is counted in `x`. If `weights` are non-empty, then index `i` stores the
 * sum of the value in `weights` at each index where the corresponding value in
 * `x` is `i`.
 *
 * Values in `x` outside of the range [0, size) are ignored.
 *
 * @param x The input int tensor, rank 1.
 * @param weights The weights tensor, must have the same shape as x, or a
 *     length-0 Tensor, in which case it acts as all weights equal to 1.
 * @param size Non-negative integer.
 *
 * @doc {heading: 'Operations', subheading: 'Reduction'}
 */
declare function bincount_<T extends Tensor1D>(x: T | TensorLike, weights: T | TensorLike, size: number): T;
declare const bincount: typeof bincount_;
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/bincount_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/boolean_mask" />
/**
 * Apply boolean mask to tensor.
 *
 * ```js
 * const tensor = tf.tensor2d([1, 2, 3, 4, 5, 6], [3, 2]);
 * const mask = tf.tensor1d([1, 0, 1], 'bool');
 * const result = await tf.booleanMaskAsync(tensor, mask);
 * result.print();
 * ```
 *
 * @param tensor N-D tensor.
 * @param mask K-D boolean tensor, K <= N and K must be known statically.
 * @param axis A 0-D int Tensor representing the axis in tensor to mask from.
 *     By default, axis is 0 which will mask from the first dimension.
 *     Otherwise K + axis <= N.
 *
 * @doc {heading: 'Tensors', subheading: 'Slicing and Joining'}
 */
declare function booleanMaskAsync_(tensor: Tensor | TensorLike, mask: Tensor | TensorLike, axis?: number): Promise<Tensor>;
declare const booleanMaskAsync: typeof booleanMaskAsync_;
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/boolean_mask_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/gradients/BroadcastTo_grad" />
declare const broadcastToGradConfig: GradConfig;

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/broadcast_args" />
/**
 * Return the shape of s0 op s1 with broadcast.
 *
 * compute r0, the broadcasted shape as a tensor.
 * s0, s1 and r0 are all integer vectors.
 *
 * This function returns the shape of the result of an operation between
 * two tensors of size s0 and s1 performed with broadcast.
 *
 * @param s0 A tensor representing a shape
 * @param s1 A tensor representing a shape
 *
 * @doc {heading: 'Tensors', subheading: 'Transformations'}
 */
declare function broadcastArgs_<R extends Rank>(s0: Tensor | TensorLike, s1: Tensor | TensorLike): Tensor<R>;
declare const broadcastArgs: typeof broadcastArgs_;
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/broadcast_args_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/broadcast_to" />
/**
 * Broadcast an array to a compatible shape NumPy-style.
 *
 * The tensor's shape is compared to the broadcast shape from end to beginning.
 * Ones are prepended to the tensor's shape until it has the same length as
 * the broadcast shape. If input.shape[i]==shape[i], the (i+1)-th axis is
 * already broadcast-compatible. If input.shape[i]==1 and shape[i]==N, then
 * the input tensor is tiled N times along that axis (using tf.tile).
 *
 * @param input The tensor that is to be broadcasted.
 * @param shape The input is to be broadcast to this shape.
 *
 * @doc {heading: 'Tensors', subheading: 'Transformations'}
 */
declare function broadcastTo_<R extends Rank>(x: Tensor | TensorLike, shape: ShapeMap[R]): Tensor<R>;
declare const broadcastTo: typeof broadcastTo_;
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/broadcast_to_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/broadcast_util" />
/**
 * Returns the dimensions in the input shape that are broadcasted to
 * produce the provided output shape.
 *
 * The returned dimensions are 0-indexed and sorted. An example:
 * inShape = [4, 1, 3]
 * outShape = [5, 4, 3, 3]
 * result = [1]. Dimension 1 (2nd dimension of input) gets broadcasted 1 => 3.
 */
declare function getBroadcastDims(inShape: number[], outShape: number[]): number[];
/**
 * Returns the axes in the output space that should be reduced to produce
 * the input space.
 */
declare function getReductionAxes(inShape: number[], outShape: number[]): number[];
declare function assertAndGetBroadcastShape(shapeA: number[], shapeB: number[]): number[];

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/broadcast_util_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/browser" />
/**
 * Creates a `tf.Tensor` from an image.
 *
 * ```js
 * const image = new ImageData(1, 1);
 * image.data[0] = 100;
 * image.data[1] = 150;
 * image.data[2] = 200;
 * image.data[3] = 255;
 *
 * tf.browser.fromPixels(image).print();
 * ```
 *
 * @param pixels The input image to construct the tensor from. The
 * supported image types are all 4-channel. You can also pass in an image
 * object with following attributes:
 * `{data: Uint8Array; width: number; height: number}`
 * @param numChannels The number of channels of the output tensor. A
 * numChannels value less than 4 allows you to ignore channels. Defaults to
 * 3 (ignores alpha channel of input image).
 *
 * @returns A Tensor3D with the shape `[height, width, numChannels]`.
 *
 * Note: fromPixels can be lossy in some cases, same image may result in
 * slightly different tensor values, if rendered by different rendering
 * engines. This means that results from different browsers, or even same
 * browser with CPU and GPU rendering engines can be different. See discussion
 * in details:
 * https://github.com/tensorflow/tfjs/issues/5482
 *
 * @doc {heading: 'Browser', namespace: 'browser', ignoreCI: true}
 */
declare function fromPixels_(pixels: PixelData | ImageData | HTMLImageElement | HTMLCanvasElement | HTMLVideoElement | ImageBitmap, numChannels?: number): Tensor3D;
/**
 * Creates a `tf.Tensor` from an image in async way.
 *
 * ```js
 * const image = new ImageData(1, 1);
 * image.data[0] = 100;
 * image.data[1] = 150;
 * image.data[2] = 200;
 * image.data[3] = 255;
 *
 * (await tf.browser.fromPixelsAsync(image)).print();
 * ```
 * This API is the async version of fromPixels. The API will first
 * check |WRAP_TO_IMAGEBITMAP| flag, and try to wrap the input to
 * imageBitmap if the flag is set to true.
 *
 * @param pixels The input image to construct the tensor from. The
 * supported image types are all 4-channel. You can also pass in an image
 * object with following attributes:
 * `{data: Uint8Array; width: number; height: number}`
 * @param numChannels The number of channels of the output tensor. A
 * numChannels value less than 4 allows you to ignore channels. Defaults to
 * 3 (ignores alpha channel of input image).
 *
 * @doc {heading: 'Browser', namespace: 'browser', ignoreCI: true}
 */
declare function fromPixelsAsync(pixels: PixelData | ImageData | HTMLImageElement | HTMLCanvasElement | HTMLVideoElement | ImageBitmap, numChannels?: number): Promise<Tensor3D>;
/**
 * Draws a `tf.Tensor` of pixel values to a byte array or optionally a
 * canvas.
 *
 * When the dtype of the input is 'float32', we assume values in the range
 * [0-1]. Otherwise, when input is 'int32', we assume values in the range
 * [0-255].
 *
 * Returns a promise that resolves when the canvas has been drawn to.
 *
 * @param img A rank-2 tensor with shape `[height, width]`, or a rank-3 tensor
 * of shape `[height, width, numChannels]`. If rank-2, draws grayscale. If
 * rank-3, must have depth of 1, 3 or 4. When depth of 1, draws
 * grayscale. When depth of 3, we draw with the first three components of
 * the depth dimension corresponding to r, g, b and alpha = 1. When depth of
 * 4, all four components of the depth dimension correspond to r, g, b, a.
 * @param canvas The canvas to draw to.
 *
 * @doc {heading: 'Browser', namespace: 'browser'}
 */
declare function toPixels(img: Tensor2D | Tensor3D | TensorLike, canvas?: HTMLCanvasElement): Promise<Uint8ClampedArray>;
declare const fromPixels: typeof fromPixels_;
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/io/browser_files" />
/**
 * IOHandlers related to files, such as browser-triggered file downloads,
 * user-selected files in browser.
 */
declare class BrowserDownloads implements IOHandler {
    private readonly modelJsonFileName;
    private readonly weightDataFileName;
    private readonly modelJsonAnchor;
    private readonly weightDataAnchor;
    static readonly URL_SCHEME = "downloads://";
    constructor(fileNamePrefix?: string);
    save(modelArtifacts: ModelArtifacts): Promise<SaveResult>;
}
declare const browserDownloadsRouter: IORouter;
/**
 * Creates an IOHandler that triggers file downloads from the browser.
 *
 * The returned `IOHandler` instance can be used as model exporting methods such
 * as `tf.Model.save` and supports only saving.
 *
 * ```js
 * const model = tf.sequential();
 * model.add(tf.layers.dense(
 *     {units: 1, inputShape: [10], activation: 'sigmoid'}));
 * const saveResult = await model.save('downloads://mymodel');
 * // This will trigger downloading of two files:
 * //   'mymodel.json' and 'mymodel.weights.bin'.
 * console.log(saveResult);
 * ```
 *
 * @param fileNamePrefix Prefix name of the files to be downloaded. For use with
 *   `tf.Model`, `fileNamePrefix` should follow either of the following two
 *   formats:
 *   1. `null` or `undefined`, in which case the default file
 *      names will be used:
 *      - 'model.json' for the JSON file containing the model topology and
 *        weights manifest.
 *      - 'model.weights.bin' for the binary file containing the binary weight
 *        values.
 *   2. A single string or an Array of a single string, as the file name prefix.
 *      For example, if `'foo'` is provided, the downloaded JSON
 *      file and binary weights file will be named 'foo.json' and
 *      'foo.weights.bin', respectively.
 * @param config Additional configuration for triggering downloads.
 * @returns An instance of `BrowserDownloads` `IOHandler`.
 *
 * @doc {
 *   heading: 'Models',
 *   subheading: 'Loading',
 *   namespace: 'io',
 *   ignoreCI: true
 * }
 */
declare function browserDownloads(fileNamePrefix?: string): IOHandler;
/**
 * Creates an IOHandler that loads model artifacts from user-selected files.
 *
 * This method can be used for loading from files such as user-selected files
 * in the browser.
 * When used in conjunction with `tf.loadLayersModel`, an instance of
 * `tf.LayersModel` (Keras-style) can be constructed from the loaded artifacts.
 *
 * ```js
 * // Note: This code snippet won't run properly without the actual file input
 * //   elements in the HTML DOM.
 *
 * // Suppose there are two HTML file input (`<input type="file" ...>`)
 * // elements.
 * const uploadJSONInput = document.getElementById('upload-json');
 * const uploadWeightsInput = document.getElementById('upload-weights');
 * const model = await tf.loadLayersModel(tf.io.browserFiles(
 *     [uploadJSONInput.files[0], uploadWeightsInput.files[0]]));
 * ```
 *
 * @param files `File`s to load from. Currently, this function supports only
 *   loading from files that contain Keras-style models (i.e., `tf.Model`s), for
 *   which an `Array` of `File`s is expected (in that order):
 *   - A JSON file containing the model topology and weight manifest.
 *   - Optionally, one or more binary files containing the binary weights.
 *     These files must have names that match the paths in the `weightsManifest`
 *     contained by the aforementioned JSON file, or errors will be thrown
 *     during loading. These weights files have the same format as the ones
 *     generated by `tensorflowjs_converter` that comes with the `tensorflowjs`
 *     Python PIP package. If no weights files are provided, only the model
 *     topology will be loaded from the JSON file above.
 * @returns An instance of `Files` `IOHandler`.
 *
 * @doc {
 *   heading: 'Models',
 *   subheading: 'Loading',
 *   namespace: 'io',
 *   ignoreCI: true
 * }
 */
declare function browserFiles(files: File[]): IOHandler;

/// <amd-module name="@tensorflow/tfjs-core/dist/io/browser_files_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/browser_util" />
/**
 * Returns a promise that resolves when a requestAnimationFrame has completed.
 *
 * On Node.js this uses setImmediate instead of requestAnimationFrame.
 *
 * This is simply a sugar method so that users can do the following:
 * `await tf.nextFrame();`
 *
 * @doc {heading: 'Performance', subheading: 'Timing'}
 */
declare function nextFrame(): Promise<void>;
{ nextFrame };

/// <amd-module name="@tensorflow/tfjs-core/dist/browser_util_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/buffer" />
/**
 * Creates an empty `tf.TensorBuffer` with the specified `shape` and `dtype`.
 *
 * The values are stored in CPU as `TypedArray`. Fill the buffer using
 * `buffer.set()`, or by modifying directly `buffer.values`.
 *
 * When done, call `buffer.toTensor()` to get an immutable `tf.Tensor` with
 * those values.
 *
 * ```js
 * // Create a buffer and set values at particular indices.
 * const buffer = tf.buffer([2, 2]);
 * buffer.set(3, 0, 0);
 * buffer.set(5, 1, 0);
 *
 * // Convert the buffer back to a tensor.
 * buffer.toTensor().print();
 * ```
 *
 * @param shape An array of integers defining the output tensor shape.
 * @param dtype The dtype of the buffer. Defaults to 'float32'.
 * @param values The values of the buffer as `TypedArray`. Defaults to
 * zeros.
 *
 * @doc {heading: 'Tensors', subheading: 'Creation'}
 */
declare function buffer<R extends Rank, D extends DataType = 'float32'>(shape: ShapeMap[R], dtype?: D, values?: DataTypeMap[D]): TensorBuffer<R, D>;

/// <amd-module name="@tensorflow/tfjs-core/dist/buffer_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-data/dist/iterators/byte_chunk_iterator" />
declare abstract class ByteChunkIterator extends LazyIterator<Uint8Array> {
    /**
     * Decode a stream of UTF8-encoded byte arrays to a stream of strings.
     *
     * The byte arrays producetd from the ByteChunkIterator on which this is
     * called will be interpreted as concatenated.  No assumptions are made about
     * the boundaries of the incoming chunks, so a multi-byte UTF8 encoding of a
     * character may span the boundary between chunks.  This naturally happens,
     * for instance, when reading fixed-size byte arrays from a file.
     */
    decodeUTF8(): StringIterator;
}

/// <amd-module name="@tensorflow/tfjs-layers/dist/callbacks" />
declare abstract class Callback extends BaseCallback {
    /** Instance of `keras.models.Model`. Reference of the model being trained. */
    model: LayersModel;
    setModel(model: Container): void;
}
interface EarlyStoppingCallbackArgs {
    /**
     * Quantity to be monitored.
     *
     * Defaults to 'val_loss'.
     */
    monitor?: string;
    /**
     * Minimum change in the monitored quantity to qualify as improvement,
     * i.e., an absolute change of less than `minDelta` will count as no
     * improvement.
     *
     * Defaults to 0.
     */
    minDelta?: number;
    /**
     * Number of epochs with no improvement after which training will be stopped.
     *
     * Defaults to 0.
     */
    patience?: number;
    /** Verbosity mode. */
    verbose?: number;
    /**
     * Mode: one of 'min', 'max', and 'auto'.
     * - In 'min' mode, training will be stopped when the quantity monitored has
     *   stopped decreasing.
     * - In 'max' mode, training will be stopped when the quantity monitored has
     *   stopped increasing.
     * - In 'auto' mode, the direction is inferred automatically from the name of
     *   the monitored quantity.
     *
     * Defaults to 'auto'.
     */
    mode?: 'auto' | 'min' | 'max';
    /**
     * Baseline value of the monitored quantity.
     *
     * If specified, training will be stopped if the model doesn't show
     * improvement over the baseline.
     */
    baseline?: number;
    /**
     * Whether to restore model weights from the epoch with the best value
     * of the monitored quantity. If `False`, the model weights obtained at the
     * last step of training are used.
     *
     * **`True` is not supported yet.**
     */
    restoreBestWeights?: boolean;
}
/**
 * A Callback that stops training when a monitored quantity has stopped
 * improving.
 */
declare class EarlyStopping extends Callback {
    protected readonly monitor: string;
    protected readonly minDelta: number;
    protected readonly patience: number;
    protected readonly baseline: number;
    protected readonly verbose: number;
    protected readonly mode: 'auto' | 'min' | 'max';
    protected monitorFunc: (currVal: number, prevVal: number) => boolean;
    private wait;
    private stoppedEpoch;
    private best;
    constructor(args?: EarlyStoppingCallbackArgs);
    onTrainBegin(logs?: Logs): Promise<void>;
    onEpochEnd(epoch: number, logs?: Logs): Promise<void>;
    onTrainEnd(logs?: Logs): Promise<void>;
    private getMonitorValue;
}
/**
 * Factory function for a Callback that stops training when a monitored
 * quantity has stopped improving.
 *
 * Early stopping is a type of regularization, and protects model against
 * overfitting.
 *
 * The following example based on fake data illustrates how this callback
 * can be used during `tf.LayersModel.fit()`:
 *
 * ```js
 * const model = tf.sequential();
 * model.add(tf.layers.dense({
 *   units: 3,
 *   activation: 'softmax',
 *   kernelInitializer: 'ones',
 *   inputShape: [2]
 * }));
 * const xs = tf.tensor2d([1, 2, 3, 4], [2, 2]);
 * const ys = tf.tensor2d([[1, 0, 0], [0, 1, 0]], [2, 3]);
 * const xsVal = tf.tensor2d([4, 3, 2, 1], [2, 2]);
 * const ysVal = tf.tensor2d([[0, 0, 1], [0, 1, 0]], [2, 3]);
 * model.compile(
 *     {loss: 'categoricalCrossentropy', optimizer: 'sgd', metrics: ['acc']});
 *
 * // Without the EarlyStopping callback, the val_acc value would be:
 * //   0.5, 0.5, 0.5, 0.5, ...
 * // With val_acc being monitored, training should stop after the 2nd epoch.
 * const history = await model.fit(xs, ys, {
 *   epochs: 10,
 *   validationData: [xsVal, ysVal],
 *   callbacks: tf.callbacks.earlyStopping({monitor: 'val_acc'})
 * });
 *
 * // Expect to see a length-2 array.
 * console.log(history.history.val_acc);
 * ```
 *
 * @doc {
 *   heading: 'Callbacks',
 *   namespace: 'callbacks'
 * }
 */
declare function earlyStopping(args?: EarlyStoppingCallbackArgs): EarlyStopping;
declare const callbacks: {
    earlyStopping: typeof earlyStopping;
};

/// <amd-module name="@tensorflow/tfjs-core/dist/public/chained_ops/cast" />
declare module '../../tensor' {
    interface Tensor<R extends Rank = Rank> {
        cast<T extends Tensor>(dtype: DataType): T;
    }
}

/// <amd-module name="@tensorflow/tfjs-core/dist/gradients/Cast_grad" />
declare const castGradConfig: GradConfig;

/// <amd-module name="@tensorflow/tfjs-layers/dist/layers/preprocessing/category_encoding" />
declare interface CategoryEncodingArgs extends LayerArgs {
    numTokens: number;
    outputMode?: OutputMode;
}
declare class CategoryEncoding extends Layer {
    /** @nocollapse */
    static className: string;
    private readonly numTokens;
    private readonly outputMode;
    constructor(args: CategoryEncodingArgs);
    getConfig(): serialization.ConfigDict;
    computeOutputShape(inputShape: Shape | Shape[]): Shape | Shape[];
    call(inputs: Tensor | Tensor[], kwargs: Kwargs): Tensor[] | Tensor;
}

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/ceil" />
/**
 * Computes ceiling of input `tf.Tensor` element-wise: `ceil(x)`
 *
 * ```js
 * const x = tf.tensor1d([.6, 1.1, -3.3]);
 *
 * x.ceil().print();  // or tf.ceil(x)
 * ```
 * @param x The input Tensor.
 *
 * @doc {heading: 'Operations', subheading: 'Basic math'}
 */
declare function ceil_<T extends Tensor>(x: T | TensorLike): T;
declare const ceil: typeof ceil_;
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/gradients/Ceil_grad" />
declare const ceilGradConfig: GradConfig;

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/ceil_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-layers/dist/layers/preprocessing/center_crop" />
declare interface CenterCropArgs extends LayerArgs {
    height: number;
    width: number;
}
declare class CenterCrop extends Layer {
    /** @nocollapse */
    static className: string;
    private readonly height;
    private readonly width;
    constructor(args: CenterCropArgs);
    centerCrop(inputs: Tensor3D | Tensor4D, hBuffer: number, wBuffer: number, height: number, width: number, inputHeight: number, inputWidth: number, dtype: DataType): Tensor | Tensor[];
    upsize(inputs: Tensor3D | Tensor4D, height: number, width: number, dtype: DataType): Tensor | Tensor[];
    call(inputs: Tensor3D | Tensor4D, kwargs: Kwargs): Tensor[] | Tensor;
    getConfig(): serialization.ConfigDict;
    computeOutputShape(inputShape: Shape | Shape[]): Shape | Shape[];
}

/// <amd-module name="@tensorflow/tfjs-core/dist/gradients/ClipByValue_grad" />
declare const clipByValueGradConfig: GradConfig;
/// <amd-module name="@tensorflow/tfjs-core/dist/ops/clip_by_value" />
/**
 * Clips values element-wise. `max(min(x, clipValueMax), clipValueMin)`
 *
 * ```js
 * const x = tf.tensor1d([-1, 2, -3, 4]);
 *
 * x.clipByValue(-2, 3).print();  // or tf.clipByValue(x, -2, 3)
 * ```
 * @param x The input tensor.
 * @param clipValueMin Lower bound of range to be clipped to.
 * @param clipValueMax Upper bound of range to be clipped to.
 *
 * @doc {heading: 'Operations', subheading: 'Basic math'}
 */
declare function clipByValue_<T extends Tensor>(x: T | TensorLike, clipValueMin: number, clipValueMax: number): T;
declare const clipByValue: typeof clipByValue_;
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/clip_by_value_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/clone" />
/**
 * Creates a new tensor with the same values and shape as the specified
 * tensor.
 *
 * ```js
 * const x = tf.tensor([1, 2]);
 *
 * x.clone().print();
 * ```
 *
 * @param x The tensor to clone.
 *
 * @doc {heading: 'Tensors', subheading: 'Creation'}
 */
declare function clone_<T extends Tensor>(x: T | TensorLike): T;
declare const clone: typeof clone_;
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/clone_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-layers/dist/backend/common" />
/**
 * Returns the value of the fuzz factor used in numeric expressions.
 */
declare function epsilon(): number;
/**
 * Sets the value of the fuzz factor used in numeric expressions.
 * @param e New value of epsilon.
 */
declare function setEpsilon(e: number): void;
/**
 * Returns the default image data format convention.
 */
declare function imageDataFormat(): DataFormat;
/// <amd-module name="@tensorflow/tfjs-core/dist/ops/complex" />
/**
 * Converts two real numbers to a complex number.
 *
 * Given a tensor `real` representing the real part of a complex number, and a
 * tensor `imag` representing the imaginary part of a complex number, this
 * operation returns complex numbers elementwise of the form [r0, i0, r1, i1],
 * where r represents the real part and i represents the imag part.
 *
 * The input tensors real and imag must have the same shape.
 *
 * ```js
 * const real = tf.tensor1d([2.25, 3.25]);
 * const imag = tf.tensor1d([4.75, 5.75]);
 * const complex = tf.complex(real, imag);
 *
 * complex.print();
 * ```
 *
 * @doc {heading: 'Tensors', subheading: 'Creation'}
 */
declare function complex_<T extends Tensor>(real: T | TensorLike, imag: T | TensorLike): T;
declare const complex: typeof complex_;
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/gradients/ComplexAbs_grad" />
declare const complexAbsGradConfig: GradConfig;
/// <amd-module name="@tensorflow/tfjs-core/dist/ops/complex_ops_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/backends/complex_util" />
/**
 * Merges real and imaginary Float32Arrays into a single complex Float32Array.
 *
 * The memory layout is interleaved as follows:
 * real: [r0, r1, r2]
 * imag: [i0, i1, i2]
 * complex: [r0, i0, r1, i1, r2, i2]
 *
 * This is the inverse of splitRealAndImagArrays.
 *
 * @param real The real values of the complex tensor values.
 * @param imag The imag values of the complex tensor values.
 * @returns A complex tensor as a Float32Array with merged values.
 */
declare function mergeRealAndImagArrays(real: Float32Array, imag: Float32Array): Float32Array;
/**
 * Splits a complex Float32Array into real and imag parts.
 *
 * The memory layout is interleaved as follows:
 * complex: [r0, i0, r1, i1, r2, i2]
 * real: [r0, r1, r2]
 * imag: [i0, i1, i2]
 *
 * This is the inverse of mergeRealAndImagArrays.
 *
 * @param complex The complex tensor values.
 * @returns An object with real and imag Float32Array components of the complex
 *     tensor.
 */
declare function splitRealAndImagArrays(complex: Float32Array): {
    real: Float32Array;
    imag: Float32Array;
};
/**
 * Extracts even indexed complex values in the given array.
 * @param complex The complex tensor values
 */
declare function complexWithEvenIndex(complex: Float32Array): {
    real: Float32Array;
    imag: Float32Array;
};
/**
 * Extracts odd indexed comple values in the given array.
 * @param complex The complex tensor values
 */
declare function complexWithOddIndex(complex: Float32Array): {
    real: Float32Array;
    imag: Float32Array;
};
/**
 * Get the map representing a complex value in the given array.
 * @param complex The complex tensor values.
 * @param index An index of the target complex value.
 */
declare function getComplexWithIndex(complex: Float32Array, index: number): {
    real: number;
    imag: number;
};
/**
 * Insert a given complex value into the TypedArray.
 * @param data The array in which the complex value is inserted.
 * @param c The complex value to be inserted.
 * @param index An index of the target complex value.
 */
declare function assignToTypedArray(data: TypedArray, real: number, imag: number, index: number): void;
/**
 * Make the list of exponent terms used by FFT.
 */
declare function exponents(n: number, inverse: boolean): {
    real: Float32Array;
    imag: Float32Array;
};
/**
 * Make the exponent term used by FFT.
 */
declare function exponent(k: number, n: number, inverse: boolean): {
    real: number;
    imag: number;
};
/// <amd-module name="@tensorflow/tfjs-core/dist/backends/complex_util_test" />
{ };
/// <amd-module name="@tensorflow/tfjs-core/dist/ops/losses/compute_weighted_loss" />

/**
 * Computes the weighted loss between two tensors.
 *
 * @param losses Tensor of shape `[batch_size, d1, ..., dN]`.
 * @param weights Tensor whose rank is either 0, or the same rank as
 *    `losses`, and must be broadcastable to `losses` (i.e., all
 *    dimensions must be either `1`, or the same as the corresponding
 *    `losses` dimension).
 *
 * @doc {heading: 'Training', subheading: 'Losses', namespace: 'losses'}
 */
declare function computeWeightedLoss_<T extends Tensor, O extends Tensor>(losses: T | TensorLike, weights?: Tensor | TensorLike, reduction?: Reduction): O;
declare const computeWeightedLoss: typeof computeWeightedLoss_;
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/losses/compute_weighted_loss_test" />
{ };
/// <amd-module name="@tensorflow/tfjs-core/dist/ops/concat" />
/**
 * Concatenates a list of `tf.Tensor`s along a given axis.
 *
 * The tensors ranks and types must match, and their sizes must match in all
 * dimensions except `axis`.
 *
 * Also available are stricter rank-specific methods that assert that
 * `tensors` are of the given rank:
 *   - `tf.concat1d`
 *   - `tf.concat2d`
 *   - `tf.concat3d`
 *   - `tf.concat4d`
 *
 * Except `tf.concat1d` (which does not have axis param), all methods have
 * same signature as this method.
 *
 * ```js
 * const a = tf.tensor1d([1, 2]);
 * const b = tf.tensor1d([3, 4]);
 * a.concat(b).print();  // or a.concat(b)
 * ```
 *
 * ```js
 * const a = tf.tensor1d([1, 2]);
 * const b = tf.tensor1d([3, 4]);
 * const c = tf.tensor1d([5, 6]);
 * tf.concat([a, b, c]).print();
 * ```
 *
 * ```js
 * const a = tf.tensor2d([[1, 2], [10, 20]]);
 * const b = tf.tensor2d([[3, 4], [30, 40]]);
 * const axis = 1;
 * tf.concat([a, b], axis).print();
 * ```
 * @param tensors A list of tensors to concatenate.
 * @param axis The axis to concatenate along. Defaults to 0 (the first dim).
 *
 * @doc {heading: 'Tensors', subheading: 'Slicing and Joining'}
 */
declare function concat_<T extends Tensor>(tensors: Array<T | TensorLike>, axis?: number): T;
declare const concat: typeof concat_;
{ };
/// <amd-module name="@tensorflow/tfjs-core/dist/ops/concat_1d" />

/**
 * Concatenates a list of`tf.Tensor1D`s along an axis. See `concat` for details.
 *
 * For example, if:
 * A: shape(3) = |r1, g1, b1|
 * B: shape(2) = |r2, g2|
 * C = tf.concat1d([A, B]) == |r1, g1, b1, r2, g2|
 *
 * @param tensors A list of`tf.Tensor`s to concatenate.
 * @return The concatenated array.
 */
declare function concat1d_(tensors: Array<Tensor1D | TensorLike>): Tensor1D;
declare const concat1d: typeof concat1d_;
{ };
/// <amd-module name="@tensorflow/tfjs-core/dist/ops/concat_2d" />

/**
 * Concatenates a list of`tf.Tensor2D`s along an axis. See `concat` for details.
 *
 * For example, if:
 * A: shape(2, 3) = | r1, g1, b1 |
 *                  | r2, g2, b2 |
 *
 * B: shape(2, 3) = | r3, g3, b3 |
 *                  | r4, g4, b4 |
 *
 * C = tf.concat2d([A, B], axis)
 *
 * if axis = 0:
 * C: shape(4, 3) = | r1, g1, b1 |
 *                  | r2, g2, b2 |
 *                  | r3, g3, b3 |
 *                  | r4, g4, b4 |
 *
 * if axis = 1:
 * C = shape(2, 6) = | r1, g1, b1, r3, g3, b3 |
 *                   | r2, g2, b2, r4, g4, b4 |
 *
 *
 * @param tensors A list of `tf.Tensor`s to concatenate.
 * @param axis The axis to concatenate along.
 * @return The concatenated array.
 */
declare function concat2d_(tensors: Array<Tensor2D | TensorLike>, axis: number): Tensor2D;
declare const concat2d: typeof concat2d_;
{ };
/// <amd-module name="@tensorflow/tfjs-core/dist/ops/concat_3d" />

/**
 * Concatenates a list of `tf.Tensor3D`s along an axis.
 * See `concat` for details.
 *
 * For example, if:
 * A: shape(2, 1, 3) = | r1, g1, b1 |
 *                     | r2, g2, b2 |
 *
 * B: shape(2, 1, 3) = | r3, g3, b3 |
 *                     | r4, g4, b4 |
 *
 * C = tf.concat3d([A, B], axis)
 *
 * if axis = 0:
 * C: shape(4, 1, 3) = | r1, g1, b1 |
 *                     | r2, g2, b2 |
 *                     | r3, g3, b3 |
 *                     | r4, g4, b4 |
 *
 * if axis = 1:
 * C: shape(2, 2, 3) = | r1, g1, b1, r3, g3, b3 |
 *                     | r2, g2, b2, r4, g4, b4 |
 *
 * if axis = 2:
 * C = shape(2, 1, 6) = | r1, g1, b1, r3, g3, b3 |
 *                      | r2, g2, b2, r4, g4, b4 |
 *
 * @param tensors A list of`tf.Tensor`s to concatenate.
 * @param axis The axis to concate along.
 * @return The concatenated array.
 */
declare function concat3d_(tensors: Array<Tensor3D | TensorLike>, axis: number): Tensor3D;
declare const concat3d: typeof concat3d_;
{ };
/// <amd-module name="@tensorflow/tfjs-core/dist/ops/concat_4d" />

/**
 * Concatenates a list of `tf.Tensor4D`s along an axis.
 * See `concat` for details.
 *
 * @param tensors A list of `tf.Tensor`s to concatenate.
 * @param axis The axis to concate along.
 * @return The concatenated array.
 */
declare function concat4d_(tensors: Array<Tensor4D | TensorLike>, axis: number): Tensor4D;
declare const concat4d: typeof concat4d_;
{ };
/// <amd-module name="@tensorflow/tfjs-core/dist/gradients/Concat_grad" />
declare const concatGradConfig: GradConfig;

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/concat_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/concat_util" />
declare function assertParamsConsistent(shapes: number[][], axis: number): void;
declare function computeOutShape(shapes: number[][], axis: number): number[];

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/concat_util_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/confusion_matrix" />
/**
 * Computes the confusion matrix from true labels and predicted labels.
 *
 * ```js
 * const labels = tf.tensor1d([0, 1, 2, 1, 0], 'int32');
 * const predictions = tf.tensor1d([0, 2, 2, 1, 0], 'int32');
 * const numClasses = 3;
 * const out = tf.math.confusionMatrix(labels, predictions, numClasses);
 * out.print();
 * // Expected output matrix:
 * // [[2, 0, 0],
 * //  [0, 1, 1],
 * //  [0, 0, 1]]
 * ```
 *
 * @param labels The target labels, assumed to be 0-based integers
 *   for the classes. The shape is `[numExamples]`, where
 *   `numExamples` is the number of examples included.
 * @param predictions The predicted classes, assumed to be
 *   0-based integers for the classes. Must have the same shape as `labels`.
 * @param numClasses Number of all classes, as an integer.
 *   Its value must be larger than the largest element in `labels` and
 *   `predictions`.
 * @returns The confusion matrix as a int32-type 2D tensor. The value at
 *   row `r` and column `c` is the number of times examples of actual class
 *   `r` were predicted as class `c`.
 *
 * @doc {heading: 'Operations', subheading: 'Evaluation'}
 */
declare function confusionMatrix_(labels: Tensor1D | TensorLike, predictions: Tensor1D | TensorLike, numClasses: number): Tensor2D;
declare const confusionMatrix: typeof confusionMatrix_;

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/confusion_matrix_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-layers/dist/constraints" />
/**
 * Base class for functions that impose constraints on weight values
 *
 * @doc {
 *   heading: 'Constraints',
 *   subheading: 'Classes',
 *   namespace: 'constraints'
 * }
 */
declare abstract class Constraint extends serialization.Serializable {
    abstract apply(w: Tensor): Tensor;
    getConfig(): serialization.ConfigDict;
}
interface MaxNormArgs {
    /**
     * Maximum norm for incoming weights
     */
    maxValue?: number;
    /**
     * Axis along which to calculate norms.
     *
     *  For instance, in a `Dense` layer the weight matrix
     *  has shape `[inputDim, outputDim]`,
     *  set `axis` to `0` to constrain each weight vector
     *  of length `[inputDim,]`.
     *  In a `Conv2D` layer with `dataFormat="channels_last"`,
     *  the weight tensor has shape
     *  `[rows, cols, inputDepth, outputDepth]`,
     *  set `axis` to `[0, 1, 2]`
     *  to constrain the weights of each filter tensor of size
     *  `[rows, cols, inputDepth]`.
     */
    axis?: number;
}
declare class MaxNorm extends Constraint {
    /** @nocollapse */
    static readonly className = "MaxNorm";
    private maxValue;
    private axis;
    private readonly defaultMaxValue;
    private readonly defaultAxis;
    constructor(args: MaxNormArgs);
    apply(w: Tensor): Tensor;
    getConfig(): serialization.ConfigDict;
}
interface UnitNormArgs {
    /**
     * Axis along which to calculate norms.
     *
     * For instance, in a `Dense` layer the weight matrix
     * has shape `[inputDim, outputDim]`,
     * set `axis` to `0` to constrain each weight vector
     * of length `[inputDim,]`.
     * In a `Conv2D` layer with `dataFormat="channels_last"`,
     * the weight tensor has shape
     * `[rows, cols, inputDepth, outputDepth]`,
     * set `axis` to `[0, 1, 2]`
     * to constrain the weights of each filter tensor of size
     * `[rows, cols, inputDepth]`.
     */
    axis?: number;
}
declare class UnitNorm extends Constraint {
    /** @nocollapse */
    static readonly className = "UnitNorm";
    private axis;
    private readonly defaultAxis;
    constructor(args: UnitNormArgs);
    apply(w: Tensor): Tensor;
    getConfig(): serialization.ConfigDict;
}
declare class NonNeg extends Constraint {
    /** @nocollapse */
    static readonly className = "NonNeg";
    apply(w: Tensor): Tensor;
}
interface MinMaxNormArgs {
    /**
     * Minimum norm for incoming weights
     */
    minValue?: number;
    /**
     * Maximum norm for incoming weights
     */
    maxValue?: number;
    /**
     * Axis along which to calculate norms.
     * For instance, in a `Dense` layer the weight matrix
     * has shape `[inputDim, outputDim]`,
     * set `axis` to `0` to constrain each weight vector
     * of length `[inputDim,]`.
     * In a `Conv2D` layer with `dataFormat="channels_last"`,
     * the weight tensor has shape
     * `[rows, cols, inputDepth, outputDepth]`,
     * set `axis` to `[0, 1, 2]`
     * to constrain the weights of each filter tensor of size
     * `[rows, cols, inputDepth]`.
     */
    axis?: number;
    /**
     * Rate for enforcing the constraint: weights will be rescaled to yield:
     * `(1 - rate) * norm + rate * norm.clip(minValue, maxValue)`.
     * Effectively, this means that rate=1.0 stands for strict
     * enforcement of the constraint, while rate<1.0 means that
     * weights will be rescaled at each step to slowly move
     * towards a value inside the desired interval.
     */
    rate?: number;
}
declare class MinMaxNorm extends Constraint {
    /** @nocollapse */
    static readonly className = "MinMaxNorm";
    private minValue;
    private maxValue;
    private rate;
    private axis;
    private readonly defaultMinValue;
    private readonly defaultMaxValue;
    private readonly defaultRate;
    private readonly defaultAxis;
    constructor(args: MinMaxNormArgs);
    apply(w: Tensor): Tensor;
    getConfig(): serialization.ConfigDict;
}
/** @docinline */
declare type ConstraintIdentifier = 'maxNorm' | 'minMaxNorm' | 'nonNeg' | 'unitNorm' | string;
declare const CONSTRAINT_IDENTIFIER_REGISTRY_SYMBOL_MAP: {
    [identifier in ConstraintIdentifier]: string;
};
declare function serializeConstraint(constraint: Constraint): serialization.ConfigDictValue;
declare function deserializeConstraint(config: serialization.ConfigDict, customObjects?: serialization.ConfigDict): Constraint;
declare function getConstraint(identifier: ConstraintIdentifier | serialization.ConfigDict | Constraint): Constraint;

/// <amd-module name="@tensorflow/tfjs-layers/dist/keras_format/constraint_config" />
declare type MaxNormConfig = {
    max_value?: number;
    axis?: number;
};
declare type MaxNormSerialization = BaseSerialization<'MaxNorm', MaxNormConfig>;
declare type UnitNormConfig = {
    axis?: number;
};
declare type UnitNormSerialization = BaseSerialization<'UnitNorm', UnitNormConfig>;
declare type NonNegSerialization = BaseSerialization<'NonNeg', null>;
declare type MinMaxNormConfig = {
    min_value?: number;
    max_value?: number;
    axis?: number;
    rate?: number;
};
declare type MinMaxNormSerialization = BaseSerialization<'MinMaxNorm', MinMaxNormConfig>;
declare type ConstraintSerialization = MaxNormSerialization | NonNegSerialization | UnitNormSerialization | MinMaxNormSerialization;
declare type ConstraintClassName = ConstraintSerialization['class_name'];
/**
 * A string array of valid Constraint class names.
 *
 * This is guaranteed to match the `ConstraintClassName` union type.
 */
declare const constraintClassNames: ConstraintClassName[];

/// <amd-module name="@tensorflow/tfjs-layers/dist/engine/container" />
/** Constructor config for Container. */
interface ContainerArgs {
    inputs: SymbolicTensor | SymbolicTensor[];
    outputs: SymbolicTensor | SymbolicTensor[];
    name?: string;
}
/**
 * A Container is a directed acyclic graph of layers.
 *
 * It is the topological form of a "model". A LayersModel
 * is simply a Container with added training routines.
 *
 */
declare abstract class Container extends Layer {
    inputs: SymbolicTensor[];
    outputs: SymbolicTensor[];
    inputLayers: Layer[];
    inputLayersNodeIndices: number[];
    inputLayersTensorIndices: number[];
    outputLayers: Layer[];
    outputLayersNodeIndices: number[];
    outputLayersTensorIndices: number[];
    layers: Layer[];
    layersByDepth: {
        [depth: string]: Layer[];
    };
    nodesByDepth: {
        [depth: string]: Node[];
    };
    internalContainerRefs: Container[];
    containerNodes: Set<string>;
    inputNames: string[];
    outputNames: string[];
    feedInputShapes: Shape[];
    protected internalInputShapes: Shape[];
    protected internalOutputShapes: Shape[];
    protected feedInputNames: string[];
    protected feedOutputNames: string[];
    constructor(args: ContainerArgs);
    protected assertNotDisposed(): void;
    /**
     * Attempt to dispose a LayersModel's weights.
     *
     * This method decrease the reference count of the LayersModel object by 1.
     *
     * A LayersModel is reference-counted. Its reference count is incremented by 1
     * when it is first constructed and when it is used as a Layer of another
     * LayersModel.
     *
     * If the reference count of a LayersModel becomes 0, the `dispose` method of
     * all its constituent `Layer`s will be called.
     *
     * Note: If the reference count is greater than 0 after the decrement, the
     * `dispose` method of its constituent `Layer`s will *not* be called.
     *
     * After a LayersModel is disposed, it cannot be used in calls such as
     * 'predict`, `evaluate` or `fit` anymore.
     *
     * @returns A DisposeResult Object with the following fields:
     *   - refCountAfterDispose: The reference count of the LayersModel after this
     *     `dispose()` call.
     *   - numDisposedVariables: Number of `tf.Variable`s (i.e., weights) disposed
     *     during this `dispose()` call.
     * @throws {Error} If the layer is not built yet, or if the LayersModel has
     *   already been disposed.
     */
    dispose(): DisposeResult;
    get trainable(): boolean;
    set trainable(trainable: boolean);
    get trainableWeights(): LayerVariable[];
    get nonTrainableWeights(): LayerVariable[];
    get weights(): LayerVariable[];
    /**
     * Loads all layer weights from a JSON object.
     *
     * Porting Note: HDF5 weight files cannot be directly loaded in JavaScript /
     *   TypeScript. The utility script at `scripts/pykeras.py` offers means
     *   to convert them into JSON strings compatible with this method.
     * Porting Note: TensorFlow.js Layers supports only loading by name currently.
     *
     * @param weights A JSON mapping weight names to weight values as nested
     *   arrays of numbers, or a `NamedTensorMap`, i.e., a JSON mapping weight
     *   names to `tf.Tensor` objects.
     * @param strict Require that the provided weights exactly match those
     *   required by the container.  Default: `true`.  Passing `false` means that
     *   extra weights and missing weights will be silently ignored.
     */
    loadWeights(weights: NamedTensorMap, strict?: boolean): void;
    /**
     * Util shared between different serialization methods.
     * @returns LayersModel config with Keras version information added.
     */
    protected updatedConfig(): serialization.ConfigDict;
    /**
     * Returns a JSON string containing the network configuration.
     *
     * To load a network from a JSON save file, use
     * models.modelFromJSON(jsonString);
     * @param extraJsonArgs Unused in tfjs-layers, maintained for PyKeras
     * @param returnString Whether the return value should be stringified
     *    (default: `true`).
     * @returns a JSON string if `returnString` (default), or a JSON object if
     *   `!returnString`.
     */
    toJSON(unused?: any, returnString?: boolean): string | PyJsonDict;
    /**
     * Call the model on new inputs.
     *
     * In this case `call` just reapplies all ops in the graph to the new inputs
     * (e.g. build a new computational graph from the provided inputs).
     *
     * @param inputs A tensor or list of tensors.
     * @param mask A mask or list of masks. A mask can be either a tensor or null
     *   (no mask).
     *
     * @return A tensor if there is a single output, or a list of tensors if there
     *   are more than one outputs.
     */
    call(inputs: Tensor | Tensor[], kwargs: Kwargs): Tensor | Tensor[];
    /**
     * Computes an output mask tensor.
     *
     * @param inputs Tensor or list of tensors.
     * @param mask Tensor or list of tensors.
     *
     * @return null or a tensor (or list of tensors, one per output tensor of the
     * layer).
     */
    computeMask(inputs: Tensor | Tensor[], mask?: Tensor | Tensor[]): Tensor | Tensor[];
    /**
     * Computes the output shape of the layer.
     *
     * Assumes that the layer will be built to match that input shape provided.
     *
     * @param inputShape A shape (tuple of integers) or a list of shape tuples
     *   (one per output tensor of the layer). Shape tuples can include null for
     *   free dimensions, instead of an integer.
     */
    computeOutputShape(inputShape: Shape | Shape[]): Shape | Shape[];
    /**
     * Computes output tensors for new inputs.
     *
     * Note:
     *   - Expects `inputs` to be a list (potentially with 1 element).
     *
     * @param inputs List of tensors
     * @param masks List of masks (tensors or null).
     * @return Three lists: outputTensors, outputMasks, outputShapes
     */
    protected runInternalGraph(inputs: Tensor[], masks?: Tensor[]): [
        Tensor[],
        Tensor[],
        Shape[]
    ];
    /**
     * Builds a map of internal node keys to node ordering.
     * Used in serializaion a node orderings may change as unused nodes are
     * dropped. Porting Note:  This helper method was pulled out of getConfig to
     * improve readability.
     * @param layers An array of Layers in the model.
     * @returns Map of Node Keys to index order within the layer.
     */
    private buildNodeConversionMap;
    /**
     * Retrieves a layer based on either its name (unique) or index.
     *
     * Indices are based on order of horizontal graph traversal (bottom-up).
     *
     * If both `name` and `index` are specified, `index` takes precedence.
     *
     * @param name Name of layer.
     * @param index Index of layer.
     * @returns A Layer instance.
     * @throws ValueError: In case of invalid layer name or index.
     *
     * @doc {
     *    heading: 'Layers',
     *    subheading: 'Classes',
     *    namespace: 'layers',
     *    subclasses: ['LayersModel']
     * }
     */
    getLayer(name?: string, index?: number): Layer;
    /**
     * Retrieves the Container's current loss values.
     *
     * Used for regularizers during training.
     */
    calculateLosses(): Scalar[];
    getConfig(): serialization.ConfigDict;
    /**
     * Instantiates a LayersModel from its config (output of `get_config()`).
     * @param cls the class to create
     * @param config LayersModel config dictionary.
     * @param customObjects An optional dictionary of custom objects.
     * @param fastWeightInit Optional flag to use fast weight initialization
     *   during deserialization. This is applicable to cases in which
     *   the initialization will be immediately overwritten by loaded weight
     *   values. Default: `false`.
     * @returns A LayersModel instance.
     * @throws ValueError: In case of improperly formatted config dict.
     */
    /** @nocollapse */
    static fromConfig<T extends serialization.Serializable>(cls: serialization.SerializableConstructor<T>, config: serialization.ConfigDict, customObjects?: serialization.ConfigDict, fastWeightInit?: boolean): T;
    /**
     * Determine whether the container is stateful.
     *
     * Porting Note: this is the equivalent of the stateful @property of
     *   the Container class in PyKeras.
     */
    get stateful(): boolean;
    /**
     * Reset the state of all stateful constituent layers (if any).
     *
     * Examples of stateful layers include RNN layers whose `stateful` property
     * is set as `true`.
     */
    resetStates(): void;
}
/// <amd-module name="@tensorflow/tfjs-core/dist/public/chained_ops/conv1d" />
declare module '../../tensor' {
    interface Tensor<R extends Rank = Rank> {
        conv1d<T extends Tensor2D | Tensor3D>(filter: Tensor3D | TensorLike3D, stride: number, pad: 'valid' | 'same' | number | ExplicitPadding, dataFormat?: 'NWC' | 'NCW', dilation?: number, dimRoundingMode?: 'floor' | 'round' | 'ceil'): T;
    }
}

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/conv1d_test" />
{ };
/// <amd-module name="@tensorflow/tfjs-core/dist/ops/conv2d" />
/**
 * Computes a 2D convolution over the input x.
 *
 * @param x The input tensor, of rank 4 or rank 3, of shape
 *     `[batch, height, width, inChannels]`. If rank 3, batch of 1 is
 * assumed.
 * @param filter The filter, rank 4, of shape
 *     `[filterHeight, filterWidth, inDepth, outDepth]`.
 * @param strides The strides of the convolution: `[strideHeight,
 * strideWidth]`.
 * @param pad The type of padding algorithm.
 *    - `same` and stride 1: output will be of same size as input,
 *       regardless of filter size.
 *    - `valid`: output will be smaller than input if filter is larger
 *       than 1x1.
 *   - For more info, see this guide:
 *     [https://www.tensorflow.org/api_docs/python/tf/nn/convolution](
 *          https://www.tensorflow.org/api_docs/python/tf/nn/convolution)
 * @param dataFormat: An optional string from: "NHWC", "NCHW". Defaults to
 *     "NHWC". Specify the data format of the input and output data. With the
 *     default format "NHWC", the data is stored in the order of: [batch,
 *     height, width, channels].
 * @param dilations The dilation rates: `[dilationHeight, dilationWidth]`
 *     in which we sample input values across the height and width dimensions
 *     in atrous convolution. Defaults to `[1, 1]`. If `dilations` is a single
 *     number, then `dilationHeight == dilationWidth`. If it is greater than
 *     1, then all values of `strides` must be 1.
 * @param dimRoundingMode A string from: 'ceil', 'round', 'floor'. If none is
 *     provided, it will default to truncate.
 *
 * @doc {heading: 'Operations', subheading: 'Convolution'}
 */
declare function conv2d_<T extends Tensor3D | Tensor4D>(x: T | TensorLike, filter: Tensor4D | TensorLike, strides: [number, number] | number, pad: 'valid' | 'same' | number | conv_util.ExplicitPadding, dataFormat?: 'NHWC' | 'NCHW', dilations?: [number, number] | number, dimRoundingMode?: 'floor' | 'round' | 'ceil'): T;
declare const conv2d: typeof conv2d_;
{ };
/// <amd-module name="@tensorflow/tfjs-core/dist/gradients/Conv2DBackpropInput_grad" />
declare const conv2DBackpropInputGradConfig: GradConfig;
/// <amd-module name="@tensorflow/tfjs-core/dist/ops/conv2d_backprop_filter" />
/**
 * Computes the derivative of the filter of a 2D convolution.
 *
 * @param x The input tensor, of rank 4 or rank 3 of shape
 *     [batch, height, width, inChannels]. If rank 3, batch of 1 is assumed.
 * @param dy The dy image, of rank 4 or rank 3, of shape
 *     [batch, height, width, outDepth]. If rank 3, batch of 1 is assumed.
 * @param filterShape The shape of the filter, length 4,
 *     [filterHeight, filterWidth, inDepth, outDepth].
 * @param strides The strides of the convolution: [strideHeight,
 * strideWidth].
 * @param pad A string from: 'same', 'valid'. The type of padding algorithm
 *     used in the forward prop of the op.
 * @param dataFormat: An optional string from: "NHWC", "NCHW". Defaults to
 *     "NHWC". Specify the data format of the input and output data. With the
 *     default format "NHWC", the data is stored in the order of: [batch,
 *     height, width, channels].
 * @param dimRoundingMode A string from: 'ceil', 'round', 'floor'. If none is
 *     provided, it will default to truncate.
 */
declare function conv2DBackpropFilter_<T extends Tensor3D | Tensor4D>(x: T, dy: T, filterShape: [number, number, number, number], strides: [number, number] | number, pad: 'valid' | 'same' | number | conv_util.ExplicitPadding, dataFormat?: 'NHWC' | 'NCHW', dimRoundingMode?: 'floor' | 'round' | 'ceil'): Tensor4D;
declare const conv2DBackpropFilter: typeof conv2DBackpropFilter_;
{ };
/// <amd-module name="@tensorflow/tfjs-core/dist/ops/conv2d_backprop_input" />
/**
 * Computes the derivative of the input of a 2D convolution.
 *
 * @param xShape The shape of the input: [batch, height, width, inDepth].
 * If length of 3, batch of 1 is assumed.
 * @param dy The derivative of the output, of rank 4 or rank 3 of shape
 *   `[batch, outHeight, outWidth, outDepth]`. If rank 3, batch of 1 is
 * assumed.
 * @param filter The filter, rank 4, of shape
 *     `[filterHeight, filterWidth, inDepth, outDepth]`.
 * @param strides The strides of the convolution: `[strideHeight,
 * strideWidth]`.
 * @param pad The type of padding algorithm used:
 *    - `same` and stride 1: output will be of same size as input,
 *       regardless of filter size.
 *    - `valid`: output will be smaller than input if filter is larger
 *       than 1x1.
 * @param dataFormat: An optional string from: "NHWC", "NCHW". Defaults to
 *     "NHWC". Specify the data format of the input and output data. With the
 *     default format "NHWC", the data is stored in the order of: [batch,
 *     height, width, channels].
 * @param dimRoundingMode A string from: 'ceil', 'round', 'floor'. If none is
 *     provided, it will default to truncate.
 */
declare function conv2DBackpropInput_<T extends Tensor3D | Tensor4D>(xShape: [number, number, number, number] | [number, number, number], dy: T, filter: Tensor4D, strides: [number, number] | number, pad: 'valid' | 'same' | number | conv_util.ExplicitPadding, dataFormat?: 'NHWC' | 'NCHW', dimRoundingMode?: 'floor' | 'round' | 'ceil'): T;
declare const conv2DBackpropInput: typeof conv2DBackpropInput_;
{ };
/// <amd-module name="@tensorflow/tfjs-core/dist/gradients/Conv2D_grad" />
declare const conv2DGradConfig: GradConfig;

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/conv2d_separable_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/conv2d_test" />
{ };
/// <amd-module name="@tensorflow/tfjs-core/dist/public/chained_ops/conv2d_transpose" />
declare module '../../tensor' {
    interface Tensor<R extends Rank = Rank> {
        conv2dTranspose<T extends Tensor3D | Tensor4D>(filter: Tensor4D | TensorLike4D, outputShape: [number, number, number, number] | [number, number, number], strides: [number, number] | number, pad: 'valid' | 'same' | number, dimRoundingMode?: 'floor' | 'round' | 'ceil'): T;
    }
}

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/conv2d_transpose_test" />
{ };
/// <amd-module name="@tensorflow/tfjs-core/dist/ops/conv3d" />
/**
 * Computes a 3D convolution over the input x.
 *
 * @param x The input tensor, of rank 5 or rank 4, of shape
 *     `[batch, depth, height, width, channels]`. If rank 4,
 * batch of 1 is assumed.
 * @param filter The filter, rank 5, of shape
 *     `[filterDepth, filterHeight, filterWidth, inChannels, outChannels]`.
 *      inChannels must match between input and filter.
 * @param strides The strides of the convolution: `[strideDepth, strideHeight,
 * strideWidth]`.
 * @param pad The type of padding algorithm.
 *    - `same` and stride 1: output will be of same size as input,
 *       regardless of filter size.
 *    - `valid`: output will be smaller than input if filter is larger
 *       than 1x1.
 *   - For more info, see this guide:
 *     [https://www.tensorflow.org/api_docs/python/tf/nn/convolution](
 *          https://www.tensorflow.org/api_docs/python/tf/nn/convolution)
 * @param dataFormat: An optional string from: "NDHWC", "NCDHW". Defaults to
 *     "NDHWC". Specify the data format of the input and output data. With the
 *     default format "NDHWC", the data is stored in the order of: [batch,
 *     depth, height, width, channels]. Only "NDHWC" is currently supported.
 * @param dilations The dilation rates: `[dilationDepth, dilationHeight,
 *     dilationWidth]` in which we sample input values across the height
 *     and width dimensions in atrous convolution. Defaults to `[1, 1, 1]`.
 *     If `dilations` is a single number, then
 *     `dilationDepth == dilationHeight == dilationWidth`. If it is greater
 *     than 1, then all values of `strides` must be 1.
 *
 * @doc {heading: 'Operations', subheading: 'Convolution'}
 */
declare function conv3d_<T extends Tensor4D | Tensor5D>(x: T | TensorLike, filter: Tensor5D | TensorLike, strides: [number, number, number] | number, pad: 'valid' | 'same', dataFormat?: 'NDHWC' | 'NCDHW', dilations?: [number, number, number] | number): T;
declare const conv3d: typeof conv3d_;
{ };
/// <amd-module name="@tensorflow/tfjs-core/dist/ops/conv3d_backprop_filter" />
/**
 * Computes the derivative of the filter of a 3D convolution.
 *
 * @param x The input tensor, of rank 5 or rank 4 of shape
 *     [batch, depth, height, width, inChannels]. If rank 4, batch of 1 is
 *     assumed.
 * @param dy The dy image, of rank 5 or rank 4, of shape
 *     [batch, depth, height, width, outDepth]. If rank 4, batch of 1 is
 *     assumed.
 * @param filterShape The shape of the filter, length 5,
 *     [filterDepth, filterHeight, filterWidth, inDepth, outDepth].
 * @param strides The strides of the convolution: [strideDepth, strideHeight,
 * strideWidth].
 * @param pad A string from: 'same', 'valid'. The type of padding algorithm
 *     used in the forward prop of the op.
 */
declare function conv3DBackpropFilter_<T extends Tensor4D | Tensor5D>(x: T, dy: T, filterShape: [number, number, number, number, number], strides: [number, number, number] | number, pad: 'valid' | 'same'): Tensor5D;
declare const conv3DBackpropFilter: typeof conv3DBackpropFilter_;
{ };
/// <amd-module name="@tensorflow/tfjs-core/dist/ops/conv3d_backprop_input" />
/**
 * Computes the derivative of the input of a 3D convolution.
 *
 * @param xShape The shape of the input: [batch, depth, height, width,
 * in_channels]. If length of 4, batch of 1 is assumed.
 * @param dy The derivative of the output, of rank 5 or rank 4 of shape
 *   `[batch, outDepth, outHeight, outWidth, in_channels]`.
 * If rank 4, batch of 1 is assumed.
 * @param filter The filter, rank 5, of shape
 *     `[filterDepth, filterHeight, filterWidth, inDepth, outDepth]`.
 * @param strides The strides of the convolution: `[strideDepth, strideHeight,
 * strideWidth]`.
 * @param pad The type of padding algorithm used:
 *    - `same` and stride 1: output will be of same size as input,
 *       regardless of filter size.
 *    - `valid`: output will be smaller than input if filter is larger
 *       than 1x1.
 */
declare function conv3DBackpropInput_<T extends Tensor4D | Tensor5D>(xShape: [
    number,
    number,
    number,
    number,
    number
] | [number, number, number, number], dy: T, filter: Tensor5D, strides: [number, number, number] | number, pad: 'valid' | 'same'): T;
declare const conv3DBackpropInput: typeof conv3DBackpropInput_;
{ };
/// <amd-module name="@tensorflow/tfjs-core/dist/gradients/Conv3D_grad" />
declare const conv3DGradConfig: GradConfig;

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/conv3d_test" />
{ };
/// <amd-module name="@tensorflow/tfjs-core/dist/ops/conv3d_transpose" />

/**
 * Computes the transposed 3D convolution of a volume, also known as a
 * deconvolution.
 *
 * @param x The input image, of rank 5 or rank 4, of shape
 *   `[batch, depth, height, width, inDepth]`. If rank 4, batch of 1 is assumed.
 * @param filter The filter, rank 4, of shape
 *     `[depth, filterHeight, filterWidth, outDepth, inDepth]`.
 *     `inDepth` must match `inDepth` in `x`.
 * @param outputShape Output shape, of rank 5 or rank 4:
 *     `[batch, depth, height, width, outDepth]`. If rank 3, batch of 1 is
 *    assumed.
 * @param strides The strides of the original convolution:
 *     `[strideDepth, strideHeight, strideWidth]`.
 * @param pad  The type of padding algorithm used in the non-transpose version
 *    of the op.
 *
 * @doc {heading: 'Operations', subheading: 'Convolution'}
 */
declare function conv3dTranspose_<T extends Tensor4D | Tensor5D>(x: T | TensorLike, filter: Tensor5D | TensorLike, outputShape: [
    number,
    number,
    number,
    number,
    number
] | [number, number, number, number], strides: [number, number, number] | number, pad: 'valid' | 'same'): T;
declare const conv3dTranspose: typeof conv3dTranspose_;
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/conv3d_transpose_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-layers/dist/layers/convolutional" />
/**
 * Transpose and cast the input before the conv2d.
 * @param x Input image tensor.
 * @param dataFormat
 */
declare function preprocessConv2DInput(x: Tensor, dataFormat: DataFormat): Tensor;
/**
 * Transpose and cast the input before the conv3d.
 * @param x Input image tensor.
 * @param dataFormat
 */
declare function preprocessConv3DInput(x: Tensor, dataFormat: DataFormat): Tensor;
/**
 * 1D-convolution with bias added.
 *
 * Porting Note: This function does not exist in the Python Keras backend.
 *   It is exactly the same as `conv2d`, except the added `bias`.
 *
 * @param x Input tensor, rank-3, of shape `[batchSize, width, inChannels]`.
 * @param kernel Kernel, rank-3, of shape `[filterWidth, inDepth, outDepth]`.
 * @param bias Bias, rank-3, of shape `[outDepth]`.
 * @param strides
 * @param padding Padding mode.
 * @param dataFormat Data format.
 * @param dilationRate
 * @returns The result of the 1D convolution.
 * @throws ValueError, if `x`, `kernel` or `bias` is not of the correct rank.
 */
declare function conv1dWithBias(x: Tensor, kernel: Tensor, bias: Tensor, strides?: number, padding?: string, dataFormat?: DataFormat, dilationRate?: number): Tensor;
/**
 * 1D-convolution.
 *
 * @param x Input tensor, rank-3, of shape `[batchSize, width, inChannels]`.
 * @param kernel Kernel, rank-3, of shape `[filterWidth, inDepth, outDepth]`.s
 * @param strides
 * @param padding Padding mode.
 * @param dataFormat Data format.
 * @param dilationRate
 * @returns The result of the 1D convolution.
 * @throws ValueError, if `x`, `kernel` or `bias` is not of the correct rank.
 */
declare function conv1d(x: Tensor, kernel: Tensor, strides?: number, padding?: string, dataFormat?: DataFormat, dilationRate?: number): Tensor;
/**
 * 2D Convolution
 * @param x
 * @param kernel kernel of the convolution.
 * @param strides strides array.
 * @param padding padding mode. Default to 'valid'.
 * @param dataFormat data format. Defaults to 'channelsLast'.
 * @param dilationRate dilation rate array.
 * @returns Result of the 2D pooling.
 */
declare function conv2d(x: Tensor, kernel: Tensor, strides?: number[], padding?: string, dataFormat?: DataFormat, dilationRate?: [number, number]): Tensor;
/**
 * 2D Convolution with an added bias and optional activation.
 * Note: This function does not exist in the Python Keras Backend. This function
 * is exactly the same as `conv2d`, except the added `bias`.
 */
declare function conv2dWithBiasActivation(x: Tensor, kernel: Tensor, bias: Tensor, strides?: number[], padding?: string, dataFormat?: DataFormat, dilationRate?: [number, number], activation?: fused.Activation): Tensor;
/**
 * 3D Convolution.
 * @param x
 * @param kernel kernel of the convolution.
 * @param strides strides array.
 * @param padding padding mode. Default to 'valid'.
 * @param dataFormat data format. Defaults to 'channelsLast'.
 * @param dilationRate dilation rate array.
 * @returns Result of the 3D convolution.
 */
declare function conv3d(x: Tensor, kernel: Tensor, strides?: number[], padding?: string, dataFormat?: DataFormat, dilationRate?: [number, number, number]): Tensor;
/**
 * 3D Convolution with an added bias.
 * Note: This function does not exist in the Python Keras Backend. This function
 * is exactly the same as `conv3d`, except the added `bias`.
 */
declare function conv3dWithBias(x: Tensor, kernel: Tensor, bias: Tensor, strides?: number[], padding?: string, dataFormat?: DataFormat, dilationRate?: [number, number, number]): Tensor;
/**
 * Base LayerConfig for depthwise and non-depthwise convolutional layers.
 */
declare interface BaseConvLayerArgs extends LayerArgs {
    /**
     * The dimensions of the convolution window. If kernelSize is a number, the
     * convolutional window will be square.
     */
    kernelSize: number | number[];
    /**
     * The strides of the convolution in each dimension. If strides is a number,
     * strides in both dimensions are equal.
     *
     * Specifying any stride value != 1 is incompatible with specifying any
     * `dilationRate` value != 1.
     */
    strides?: number | number[];
    /**
     * Padding mode.
     */
    padding?: PaddingMode;
    /**
     * Format of the data, which determines the ordering of the dimensions in
     * the inputs.
     *
     * `channels_last` corresponds to inputs with shape
     *   `(batch, ..., channels)`
     *
     *  `channels_first` corresponds to inputs with shape `(batch, channels,
     * ...)`.
     *
     * Defaults to `channels_last`.
     */
    dataFormat?: DataFormat;
    /**
     * The dilation rate to use for the dilated convolution in each dimension.
     * Should be an integer or array of two or three integers.
     *
     * Currently, specifying any `dilationRate` value != 1 is incompatible with
     * specifying any `strides` value != 1.
     */
    dilationRate?: number | [number] | [number, number] | [number, number, number];
    /**
     * Activation function of the layer.
     *
     * If you don't specify the activation, none is applied.
     */
    activation?: ActivationIdentifier;
    /**
     * Whether the layer uses a bias vector. Defaults to `true`.
     */
    useBias?: boolean;
    /**
     * Initializer for the convolutional kernel weights matrix.
     */
    kernelInitializer?: InitializerIdentifier | Initializer;
    /**
     * Initializer for the bias vector.
     */
    biasInitializer?: InitializerIdentifier | Initializer;
    /**
     * Constraint for the convolutional kernel weights.
     */
    kernelConstraint?: ConstraintIdentifier | Constraint;
    /**
     * Constraint for the bias vector.
     */
    biasConstraint?: ConstraintIdentifier | Constraint;
    /**
     * Regularizer function applied to the kernel weights matrix.
     */
    kernelRegularizer?: RegularizerIdentifier | Regularizer;
    /**
     * Regularizer function applied to the bias vector.
     */
    biasRegularizer?: RegularizerIdentifier | Regularizer;
    /**
     * Regularizer function applied to the activation.
     */
    activityRegularizer?: RegularizerIdentifier | Regularizer;
}
/**
 * LayerConfig for non-depthwise convolutional layers.
 * Applies to non-depthwise convolution of all ranks (e.g, Conv1D, Conv2D,
 * Conv3D).
 */
declare interface ConvLayerArgs extends BaseConvLayerArgs {
    /**
     * The dimensionality of the output space (i.e. the number of filters in the
     * convolution).
     */
    filters: number;
}
/**
 * Abstract convolution layer.
 */
declare abstract class BaseConv extends Layer {
    protected readonly rank: number;
    protected readonly kernelSize: number[];
    protected readonly strides: number[];
    protected readonly padding: PaddingMode;
    protected readonly dataFormat: DataFormat;
    protected readonly activation: Activation;
    protected readonly useBias: boolean;
    protected readonly dilationRate: number[];
    protected readonly biasInitializer?: Initializer;
    protected readonly biasConstraint?: Constraint;
    protected readonly biasRegularizer?: Regularizer;
    protected bias: LayerVariable;
    readonly DEFAULT_KERNEL_INITIALIZER: InitializerIdentifier;
    readonly DEFAULT_BIAS_INITIALIZER: InitializerIdentifier;
    constructor(rank: number, args: BaseConvLayerArgs);
    protected static verifyArgs(args: BaseConvLayerArgs): void;
    getConfig(): serialization.ConfigDict;
}
/**
 * Abstract nD convolution layer.  Ancestor of convolution layers which reduce
 * across channels, i.e., Conv1D and Conv2D, but not DepthwiseConv2D.
 */
declare abstract class Conv extends BaseConv {
    protected readonly filters: number;
    protected kernel: LayerVariable;
    protected readonly kernelInitializer?: Initializer;
    protected readonly kernelConstraint?: Constraint;
    protected readonly kernelRegularizer?: Regularizer;
    constructor(rank: number, args: ConvLayerArgs);
    build(inputShape: Shape | Shape[]): void;
    call(inputs: Tensor | Tensor[], kwargs: Kwargs): Tensor | Tensor[];
    computeOutputShape(inputShape: Shape | Shape[]): Shape | Shape[];
    getConfig(): serialization.ConfigDict;
    protected static verifyArgs(args: ConvLayerArgs): void;
}
declare class Conv2D extends Conv {
    /** @nocollapse */
    static className: string;
    constructor(args: ConvLayerArgs);
    getConfig(): serialization.ConfigDict;
    protected static verifyArgs(args: ConvLayerArgs): void;
}
declare class Conv3D extends Conv {
    /** @nocollapse */
    static className: string;
    constructor(args: ConvLayerArgs);
    getConfig(): serialization.ConfigDict;
    protected static verifyArgs(args: ConvLayerArgs): void;
}
declare class Conv2DTranspose extends Conv2D {
    /** @nocollapse */
    static className: string;
    constructor(args: ConvLayerArgs);
    build(inputShape: Shape | Shape[]): void;
    call(inputs: Tensor | Tensor[], kwargs: Kwargs): Tensor | Tensor[];
    computeOutputShape(inputShape: Shape | Shape[]): Shape | Shape[];
    getConfig(): serialization.ConfigDict;
}
declare class Conv3DTranspose extends Conv3D {
    /** @nocollapse */
    static className: string;
    constructor(args: ConvLayerArgs);
    build(inputShape: Shape | Shape[]): void;
    call(inputs: Tensor | Tensor[], kwargs: Kwargs): Tensor | Tensor[];
    computeOutputShape(inputShape: Shape | Shape[]): Shape | Shape[];
    getConfig(): serialization.ConfigDict;
}
declare interface SeparableConvLayerArgs extends ConvLayerArgs {
    /**
     * The number of depthwise convolution output channels for each input
     * channel.
     * The total number of depthwise convolution output channels will be equal
     * to `filtersIn * depthMultiplier`. Default: 1.
     */
    depthMultiplier?: number;
    /**
     * Initializer for the depthwise kernel matrix.
     */
    depthwiseInitializer?: InitializerIdentifier | Initializer;
    /**
     * Initializer for the pointwise kernel matrix.
     */
    pointwiseInitializer?: InitializerIdentifier | Initializer;
    /**
     * Regularizer function applied to the depthwise kernel matrix.
     */
    depthwiseRegularizer?: RegularizerIdentifier | Regularizer;
    /**
     * Regularizer function applied to the pointwise kernel matrix.
     */
    pointwiseRegularizer?: RegularizerIdentifier | Regularizer;
    /**
     * Constraint function applied to the depthwise kernel matrix.
     */
    depthwiseConstraint?: ConstraintIdentifier | Constraint;
    /**
     * Constraint function applied to the pointwise kernel matrix.
     */
    pointwiseConstraint?: ConstraintIdentifier | Constraint;
}
declare class SeparableConv extends Conv {
    /** @nocollapse */
    static className: string;
    readonly depthMultiplier: number;
    protected readonly depthwiseInitializer?: Initializer;
    protected readonly depthwiseRegularizer?: Regularizer;
    protected readonly depthwiseConstraint?: Constraint;
    protected readonly pointwiseInitializer?: Initializer;
    protected readonly pointwiseRegularizer?: Regularizer;
    protected readonly pointwiseConstraint?: Constraint;
    readonly DEFAULT_DEPTHWISE_INITIALIZER: InitializerIdentifier;
    readonly DEFAULT_POINTWISE_INITIALIZER: InitializerIdentifier;
    protected depthwiseKernel: LayerVariable;
    protected pointwiseKernel: LayerVariable;
    constructor(rank: number, config?: SeparableConvLayerArgs);
    build(inputShape: Shape | Shape[]): void;
    call(inputs: Tensor | Tensor[], kwargs: Kwargs): Tensor | Tensor[];
    getConfig(): serialization.ConfigDict;
}
declare class SeparableConv2D extends SeparableConv {
    /** @nocollapse */
    static className: string;
    constructor(args?: SeparableConvLayerArgs);
}
declare class Conv1D extends Conv {
    /** @nocollapse */
    static className: string;
    constructor(args: ConvLayerArgs);
    getConfig(): serialization.ConfigDict;
    protected static verifyArgs(args: ConvLayerArgs): void;
}
declare interface Cropping2DLayerArgs extends LayerArgs {
    /**
     * Dimension of the cropping along the width and the height.
     * - If integer: the same symmetric cropping
     *  is applied to width and height.
     * - If list of 2 integers:
     *   interpreted as two different
     *   symmetric cropping values for height and width:
     *   `[symmetric_height_crop, symmetric_width_crop]`.
     * - If a list of 2 lists of 2 integers:
     *   interpreted as
     *   `[[top_crop, bottom_crop], [left_crop, right_crop]]`
     */
    cropping: number | [number, number] | [[number, number], [number, number]];
    /**
     * Format of the data, which determines the ordering of the dimensions in
     * the inputs.
     *
     * `channels_last` corresponds to inputs with shape
     *   `(batch, ..., channels)`
     *
     * `channels_first` corresponds to inputs with shape
     *   `(batch, channels, ...)`
     *
     * Defaults to `channels_last`.
     */
    dataFormat?: DataFormat;
}
declare class Cropping2D extends Layer {
    /** @nocollapse */
    static className: string;
    protected readonly cropping: [[number, number], [number, number]];
    protected readonly dataFormat: DataFormat;
    constructor(args: Cropping2DLayerArgs);
    computeOutputShape(inputShape: Shape): Shape;
    call(inputs: Tensor | Tensor[], kwargs: Kwargs): Tensor | Tensor[];
    getConfig(): serialization.ConfigDict;
}
declare interface UpSampling2DLayerArgs extends LayerArgs {
    /**
     * The upsampling factors for rows and columns.
     *
     * Defaults to `[2, 2]`.
     */
    size?: number[];
    /**
     * Format of the data, which determines the ordering of the dimensions in
     * the inputs.
     *
     * `"channelsLast"` corresponds to inputs with shape
     *   `[batch, ..., channels]`
     *
     *  `"channelsFirst"` corresponds to inputs with shape `[batch, channels,
     * ...]`.
     *
     * Defaults to `"channelsLast"`.
     */
    dataFormat?: DataFormat;
    /**
     * The interpolation mechanism, one of `"nearest"` or `"bilinear"`, default
     * to `"nearest"`.
     */
    interpolation?: InterpolationFormat;
}
declare class UpSampling2D extends Layer {
    /** @nocollapse */
    static className: string;
    protected readonly DEFAULT_SIZE: number[];
    protected readonly size: number[];
    protected readonly dataFormat: DataFormat;
    protected readonly interpolation: InterpolationFormat;
    constructor(args: UpSampling2DLayerArgs);
    computeOutputShape(inputShape: Shape): Shape;
    call(inputs: Tensor | Tensor[], kwargs: Kwargs): Tensor | Tensor[];
    getConfig(): serialization.ConfigDict;
}

/// <amd-module name="@tensorflow/tfjs-layers/dist/layers/convolutional_depthwise" />
/**
 * 2D convolution with separable filters.
 * @param x Input tensor.
 * @param depthwiseKernel Convolution kernel for depthwise convolution.
 * @param strides Strides (Array of two integers).
 * @param padding Padding model.
 * @param dataFormat Data format.
 * @param dilationRate Array of two integers, dilation rates for the separable
 *   convolution.
 * @returns Output tensor.
 * @throws ValueError If depthwiseKernel is not a 4D array.
 */
declare function depthwiseConv2d(x: Tensor, depthwiseKernel: Tensor, strides?: [number, number], padding?: string, dataFormat?: DataFormat, dilationRate?: [number, number]): Tensor;
declare interface DepthwiseConv2DLayerArgs extends BaseConvLayerArgs {
    /**
     * An integer or Array of 2 integers, specifying the width and height of the
     * 2D convolution window. Can be a single integer to specify the same value
     * for all spatial dimensions.
     */
    kernelSize: number | [number, number];
    /**
     * The number of depthwise convolution output channels for each input
     * channel.
     * The total number of depthwise convolution output channels will be equal to
     * `filtersIn * depthMultiplier`.
     * Default: 1.
     */
    depthMultiplier?: number;
    /**
     * Initializer for the depthwise kernel matrix.
     * Default: GlorotNormal.
     */
    depthwiseInitializer?: InitializerIdentifier | Initializer;
    /**
     * Constraint for the depthwise kernel matrix.
     */
    depthwiseConstraint?: ConstraintIdentifier | Constraint;
    /**
     * Regularizer function for the depthwise kernel matrix.
     */
    depthwiseRegularizer?: RegularizerIdentifier | Regularizer;
}
declare class DepthwiseConv2D extends BaseConv {
    /** @nocollapse */
    static className: string;
    private readonly depthMultiplier;
    private readonly depthwiseInitializer;
    private readonly depthwiseConstraint;
    private readonly depthwiseRegularizer;
    private depthwiseKernel;
    constructor(args: DepthwiseConv2DLayerArgs);
    build(inputShape: Shape | Shape[]): void;
    call(inputs: Tensor | Tensor[], kwargs: Kwargs): Tensor | Tensor[];
    computeOutputShape(inputShape: Shape | Shape[]): Shape | Shape[];
    getConfig(): serialization.ConfigDict;
}

/// <amd-module name="@tensorflow/tfjs-layers/dist/keras_format/layers/convolutional_depthwise_serialization" />
interface DepthwiseConv2DLayerConfig extends BaseConvLayerConfig {
    kernel_size: number | [number, number];
    depth_multiplier?: number;
    depthwise_initializer?: InitializerSerialization;
    depthwise_constraint?: ConstraintSerialization;
    depthwise_regularizer?: RegularizerSerialization;
}
declare type DepthwiseConv2DLayerSerialization = BaseLayerSerialization<'DepthwiseConv2D', DepthwiseConv2DLayerConfig>;
declare type ConvolutionalDepthwiseLayerSerialization = DepthwiseConv2DLayerSerialization;
declare type ConvolutionalDepthwiseLayerClassName = ConvolutionalDepthwiseLayerSerialization['class_name'];
/**
 * A string array of valid ConvolutionalDepthwiseLayer class names.
 *
 * This is guaranteed to match the `ConvolutionalDepthwiseLayerClassName` union
 * type.
 */
declare const convolutionalDepthwiseLayerClassNames: ConvolutionalDepthwiseLayerClassName[];

/// <amd-module name="@tensorflow/tfjs-layers/dist/layers/convolutional_recurrent" />
declare interface ConvRNN2DCellArgs extends Omit<SimpleRNNCellLayerArgs, 'units'> {
    /**
     * The dimensionality of the output space (i.e. the number of filters in the
     * convolution).
     */
    filters: number;
    /**
     * The dimensions of the convolution window. If kernelSize is a number, the
     * convolutional window will be square.
     */
    kernelSize: number | number[];
    /**
     * The strides of the convolution in each dimension. If strides is a number,
     * strides in both dimensions are equal.
     *
     * Specifying any stride value != 1 is incompatible with specifying any
     * `dilationRate` value != 1.
     */
    strides?: number | number[];
    /**
     * Padding mode.
     */
    padding?: PaddingMode;
    /**
     * Format of the data, which determines the ordering of the dimensions in
     * the inputs.
     *
     * `channels_last` corresponds to inputs with shape
     *   `(batch, ..., channels)`
     *
     *  `channels_first` corresponds to inputs with shape `(batch, channels,
     * ...)`.
     *
     * Defaults to `channels_last`.
     */
    dataFormat?: DataFormat;
    /**
     * The dilation rate to use for the dilated convolution in each dimension.
     * Should be an integer or array of two or three integers.
     *
     * Currently, specifying any `dilationRate` value != 1 is incompatible with
     * specifying any `strides` value != 1.
     */
    dilationRate?: number | [number] | [number, number];
}
declare abstract class ConvRNN2DCell extends RNNCell {
    readonly filters: number;
    readonly kernelSize: number[];
    readonly strides: number[];
    readonly padding: PaddingMode;
    readonly dataFormat: DataFormat;
    readonly dilationRate: number[];
    readonly activation: Activation;
    readonly useBias: boolean;
    readonly kernelInitializer: Initializer;
    readonly recurrentInitializer: Initializer;
    readonly biasInitializer: Initializer;
    readonly kernelConstraint: Constraint;
    readonly recurrentConstraint: Constraint;
    readonly biasConstraint: Constraint;
    readonly kernelRegularizer: Regularizer;
    readonly recurrentRegularizer: Regularizer;
    readonly biasRegularizer: Regularizer;
    readonly dropout: number;
    readonly recurrentDropout: number;
}
declare interface ConvRNN2DLayerArgs extends BaseRNNLayerArgs, ConvRNN2DCellArgs {
}
/**
 * Base class for convolutional-recurrent layers.
 */
declare class ConvRNN2D extends RNN {
    /** @nocollapse */
    static className: string;
    readonly cell: ConvRNN2DCell;
    constructor(args: ConvRNN2DLayerArgs);
    call(inputs: Tensor | Tensor[], kwargs: Kwargs): Tensor | Tensor[];
    computeOutputShape(inputShape: Shape): Shape | Shape[];
    getInitialState(inputs: tfc.Tensor): tfc.Tensor[];
    resetStates(states?: Tensor | Tensor[], training?: boolean): void;
    protected computeSingleOutputShape(inputShape: Shape): Shape;
}
declare interface ConvLSTM2DCellArgs extends Omit<LSTMCellLayerArgs, 'units'>, ConvRNN2DCellArgs {
}
declare class ConvLSTM2DCell extends LSTMCell implements ConvRNN2DCell {
    /** @nocollapse */
    static className: string;
    readonly filters: number;
    readonly kernelSize: number[];
    readonly strides: number[];
    readonly padding: PaddingMode;
    readonly dataFormat: DataFormat;
    readonly dilationRate: number[];
    constructor(args: ConvLSTM2DCellArgs);
    build(inputShape: Shape | Shape[]): void;
    call(inputs: tfc.Tensor[], kwargs: Kwargs): tfc.Tensor[];
    getConfig(): tfc.serialization.ConfigDict;
    inputConv(x: Tensor, w: Tensor, b?: Tensor, padding?: PaddingMode): tfc.Tensor3D;
    recurrentConv(x: Tensor, w: Tensor): tfc.Tensor3D;
}
declare interface ConvLSTM2DArgs extends Omit<LSTMLayerArgs, 'units' | 'cell'>, ConvRNN2DLayerArgs {
}
declare class ConvLSTM2D extends ConvRNN2D {
    /** @nocollapse */
    static className: string;
    constructor(args: ConvLSTM2DArgs);
    /** @nocollapse */
    static fromConfig<T extends tfc.serialization.Serializable>(cls: tfc.serialization.SerializableConstructor<T>, config: tfc.serialization.ConfigDict): T;
}
{ };

/// <amd-module name="@tensorflow/tfjs-layers/dist/keras_format/layers/convolutional_serialization" />
interface BaseConvLayerConfig extends LayerConfig {
    kernel_size: number | number[];
    strides?: number | number[];
    padding?: PaddingMode;
    data_format?: DataFormatSerialization;
    dilation_rate?: number | [number] | [number, number];
    activation?: string;
    use_bias?: boolean;
    kernel_initializer?: InitializerSerialization;
    bias_initializer?: InitializerSerialization;
    kernel_constraint?: ConstraintSerialization;
    bias_constraint?: ConstraintSerialization;
    kernel_regularizer?: RegularizerSerialization;
    bias_regularizer?: RegularizerSerialization;
    activity_regularizer?: RegularizerSerialization;
}
interface ConvLayerConfig extends BaseConvLayerConfig {
    filters: number;
}
declare type Conv1DLayerSerialization = BaseLayerSerialization<'Conv1D', ConvLayerConfig>;
declare type Conv2DLayerSerialization = BaseLayerSerialization<'Conv2D', ConvLayerConfig>;
declare type Conv2DTransposeLayerSerialization = BaseLayerSerialization<'Conv2DTranspose', ConvLayerConfig>;
interface SeparableConvLayerConfig extends ConvLayerConfig {
    depth_multiplier?: number;
    depthwise_initializer?: InitializerSerialization;
    pointwise_initializer?: InitializerSerialization;
    depthwise_regularizer?: RegularizerSerialization;
    pointwise_regularizer?: RegularizerSerialization;
    depthwise_constraint?: ConstraintSerialization;
    pointwise_constraint?: ConstraintSerialization;
}
declare type SeparableConv2DLayerSerialization = BaseLayerSerialization<'SeparableConv2D', ConvLayerConfig>;
interface Cropping2DLayerConfig extends LayerConfig {
    cropping: number | [number, number] | [[number, number], [number, number]];
    data_format?: DataFormatSerialization;
}
declare type Cropping2DLayerSerialization = BaseLayerSerialization<'Cropping2D', Cropping2DLayerConfig>;
interface UpSampling2DLayerConfig extends LayerConfig {
    size?: number[];
    data_format?: DataFormatSerialization;
}
declare type UpSampling2DLayerSerialization = BaseLayerSerialization<'UpSampling2D', UpSampling2DLayerConfig>;
declare type ConvolutionalLayerSerialization = Conv1DLayerSerialization | Conv2DLayerSerialization | Conv2DTransposeLayerSerialization | SeparableConv2DLayerSerialization | Cropping2DLayerSerialization | UpSampling2DLayerSerialization;
declare type ConvolutionalLayerClassName = ConvolutionalLayerSerialization['class_name'];
/**
 * A string array of valid ConvolutionalLayer class names.
 *
 * This is guaranteed to match the `ConvolutionalLayerClassName` union type.
 */
declare const convolutionalLayerClassNames: ConvolutionalLayerClassName[];

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/conv_util" />
declare type PadType = 'SAME' | 'VALID' | 'NUMBER' | 'EXPLICIT';
declare type ExplicitPadding = [
    [number, number],
    [number, number],
    [number, number],
    [number, number]
];
declare type PadInfo = {
    top: number;
    left: number;
    right: number;
    bottom: number;
    type: PadType;
};
declare type PadInfo3D = {
    top: number;
    left: number;
    right: number;
    bottom: number;
    front: number;
    back: number;
    type: PadType;
};
/**
 * Information about the forward pass of a convolution/pooling operation.
 * It includes input and output shape, strides, filter size and padding
 * information.
 */
declare type Conv2DInfo = {
    batchSize: number;
    inHeight: number;
    inWidth: number;
    inChannels: number;
    outHeight: number;
    outWidth: number;
    outChannels: number;
    dataFormat: 'channelsFirst' | 'channelsLast';
    strideHeight: number;
    strideWidth: number;
    dilationHeight: number;
    dilationWidth: number;
    filterHeight: number;
    filterWidth: number;
    effectiveFilterHeight: number;
    effectiveFilterWidth: number;
    padInfo: PadInfo;
    inShape: [number, number, number, number];
    outShape: [number, number, number, number];
    filterShape: [number, number, number, number];
};
/**
 *
 * @param inputShape Input tensor shape is of the following dimensions:
 *     `[batch, height, width, inChannels]`.
 * @param filterShape The filter shape is of the following dimensions:
 *     `[filterHeight, filterWidth, depth]`.
 * @param strides The strides of the sliding window for each dimension of the
 *     input tensor: `[strideHeight, strideWidth]`.
 *     If `strides` is a single number,
 *     then `strideHeight == strideWidth`.
 * @param pad The type of padding algorithm.
 *    - `same` and stride 1: output will be of same size as input,
 *       regardless of filter size.
 *    - `valid`: output will be smaller than input if filter is larger
 *       than 1*1x1.
 *    - For more info, see this guide:
 *     [https://www.tensorflow.org/api_docs/python/tf/nn/convolution](
 *          https://www.tensorflow.org/api_docs/python/tf/nn/convolution)
 * @param dataFormat The data format of the input and output data.
 *     Defaults to 'NHWC'.
 * @param dilations The dilation rates: `[dilationHeight, dilationWidth]`.
 *     Defaults to `[1, 1]`. If `dilations` is a single number, then
 *     `dilationHeight == dilationWidth`.
 */
declare function computeDilation2DInfo(inputShape: [number, number, number, number], filterShape: [number, number, number], strides: number | [number, number], pad: 'same' | 'valid' | number, dataFormat: 'NHWC', dilations: number | [number, number]): Conv2DInfo;
declare function computePool2DInfo(inShape: [number, number, number, number], filterSize: [number, number] | number, strides: number | [number, number], dilations: number | [number, number], pad: 'same' | 'valid' | number | ExplicitPadding, roundingMode?: 'floor' | 'round' | 'ceil', dataFormat?: 'channelsFirst' | 'channelsLast'): Conv2DInfo;
/**
 * Computes the information for a forward pass of a pooling3D operation.
 */
declare function computePool3DInfo(inShape: [number, number, number, number, number], filterSize: number | [number, number, number], strides: number | [number, number, number], dilations: number | [number, number, number], pad: 'same' | 'valid' | number, roundingMode?: 'floor' | 'round' | 'ceil', dataFormat?: 'NDHWC' | 'NCDHW'): Conv3DInfo;
/**
 * Computes the information for a forward pass of a convolution/pooling
 * operation.
 */
declare function computeConv2DInfo(inShape: [number, number, number, number], filterShape: [number, number, number, number], strides: number | [number, number], dilations: number | [number, number], pad: 'same' | 'valid' | number | ExplicitPadding, roundingMode?: 'floor' | 'round' | 'ceil', depthwise?: boolean, dataFormat?: 'channelsFirst' | 'channelsLast'): Conv2DInfo;
/**
 * Information about the forward pass of a 3D convolution/pooling operation.
 * It includes input and output shape, strides, filter size and padding
 * information.
 */
declare type Conv3DInfo = {
    batchSize: number;
    inDepth: number;
    inHeight: number;
    inWidth: number;
    inChannels: number;
    outDepth: number;
    outHeight: number;
    outWidth: number;
    outChannels: number;
    dataFormat: 'channelsFirst' | 'channelsLast';
    strideDepth: number;
    strideHeight: number;
    strideWidth: number;
    dilationDepth: number;
    dilationHeight: number;
    dilationWidth: number;
    filterDepth: number;
    filterHeight: number;
    filterWidth: number;
    effectiveFilterDepth: number;
    effectiveFilterHeight: number;
    effectiveFilterWidth: number;
    padInfo: PadInfo3D;
    inShape: [number, number, number, number, number];
    outShape: [number, number, number, number, number];
    filterShape: [number, number, number, number, number];
};
/**
 * Computes the information for a forward pass of a 3D convolution/pooling
 * operation.
 */
declare function computeConv3DInfo(inShape: [number, number, number, number, number], filterShape: [number, number, number, number, number], strides: number | [number, number, number], dilations: number | [number, number, number], pad: 'same' | 'valid' | number, depthwise?: boolean, dataFormat?: 'channelsFirst' | 'channelsLast', roundingMode?: 'floor' | 'round' | 'ceil'): Conv3DInfo;
declare function computeDefaultPad(inputShape: [number, number] | [number, number, number, number], fieldSize: number, stride: number, dilation?: number): number;
declare function tupleValuesAreOne(param: number | number[]): boolean;
declare function eitherStridesOrDilationsAreOne(strides: number | number[], dilations: number | number[]): boolean;
declare function stridesOrDilationsArePositive(values: number | number[]): boolean;
/**
 * Convert Conv2D dataFormat from 'NHWC'|'NCHW' to
 *    'channelsLast'|'channelsFirst'
 * @param dataFormat in 'NHWC'|'NCHW' mode
 * @return dataFormat in 'channelsLast'|'channelsFirst' mode
 * @throws unknown dataFormat
 */
declare function convertConv2DDataFormat(dataFormat: 'NHWC' | 'NCHW'): 'channelsLast' | 'channelsFirst';
/**
 * Check validity of pad when using dimRoundingMode.
 * @param opDesc A string of op description
 * @param pad The type of padding algorithm.
 *   - `same` and stride 1: output will be of same size as input,
 *       regardless of filter size.
 *   - `valid` output will be smaller than input if filter is larger
 *       than 1x1.
 *   - For more info, see this guide:
 *     [https://www.tensorflow.org/api_docs/python/tf/nn/convolution](
 *          https://www.tensorflow.org/api_docs/python/tf/nn/convolution)
 * @param dimRoundingMode A string from: 'ceil', 'round', 'floor'. If none is
 *     provided, it will default to truncate.
 * @throws unknown padding parameter
 */
declare function checkPadOnDimRoundingMode(opDesc: string, pad: 'valid' | 'same' | number | ExplicitPadding, dimRoundingMode?: 'floor' | 'round' | 'ceil'): void;
{ };

/// <amd-module name="@tensorflow/tfjs-layers/dist/utils/conv_utils" />
/**
 * Transforms a single number of array of numbers into an array of numbers.
 * @param value
 * @param n: The size of the tuple to be returned.
 * @param name: Name of the parameter, used for generating error messages.
 * @returns An array of numbers.
 */
declare function normalizeArray(value: number | number[], n: number, name: string): number[];
/**
 * Determines output length of a convolution given input length.
 * @param inputLength
 * @param filterSize
 * @param padding
 * @param stride
 * @param dilation: dilation rate.
 */
declare function convOutputLength(inputLength: number, filterSize: number, padding: PaddingMode, stride: number, dilation?: number): number;
declare function deconvLength(dimSize: number, strideSize: number, kernelSize: number, padding: PaddingMode): number;

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/conv_util_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-layers/dist/layers/core" />
/**
 * TensorFlow.js Layers: Basic Layers.
 */
declare interface DropoutLayerArgs extends LayerArgs {
    /** Float between 0 and 1. Fraction of the input units to drop. */
    rate: number;
    /**
     * Integer array representing the shape of the binary dropout mask that will
     * be multiplied with the input.
     *
     * For instance, if your inputs have shape `(batchSize, timesteps, features)`
     * and you want the dropout mask to be the same for all timesteps, you can use
     * `noise_shape=(batch_size, 1, features)`.
     */
    noiseShape?: number[];
    /** An integer to use as random seed. */
    seed?: number;
}
declare class Dropout extends Layer {
    /** @nocollapse */
    static className: string;
    private readonly rate;
    private readonly noiseShape;
    private readonly seed;
    constructor(args: DropoutLayerArgs);
    protected getNoiseShape(input: Tensor): Shape;
    call(inputs: Tensor | Tensor[], kwargs: Kwargs): Tensor | Tensor[];
    getConfig(): serialization.ConfigDict;
    dispose(): DisposeResult;
}
declare interface DenseLayerArgs extends LayerArgs {
    /** Positive integer, dimensionality of the output space. */
    units: number;
    /**
     * Activation function to use.
     *
     * If unspecified, no activation is applied.
     */
    activation?: ActivationIdentifier;
    /** Whether to apply a bias. */
    useBias?: boolean;
    /**
     * Initializer for the dense kernel weights matrix.
     */
    kernelInitializer?: InitializerIdentifier | Initializer;
    /**
     * Initializer for the bias vector.
     */
    biasInitializer?: InitializerIdentifier | Initializer;
    /**
     * If specified, defines inputShape as `[inputDim]`.
     */
    inputDim?: number;
    /**
     * Constraint for the kernel weights.
     */
    kernelConstraint?: ConstraintIdentifier | Constraint;
    /**
     * Constraint for the bias vector.
     */
    biasConstraint?: ConstraintIdentifier | Constraint;
    /**
     * Regularizer function applied to the dense kernel weights matrix.
     */
    kernelRegularizer?: RegularizerIdentifier | Regularizer;
    /**
     * Regularizer function applied to the bias vector.
     */
    biasRegularizer?: RegularizerIdentifier | Regularizer;
    /**
     * Regularizer function applied to the activation.
     */
    activityRegularizer?: RegularizerIdentifier | Regularizer;
}
interface SpatialDropout1DLayerConfig extends LayerConfig {
    /** Float between 0 and 1. Fraction of the input units to drop. */
    rate: number;
    /** An integer to use as random seed. */
    seed?: number;
}
declare class SpatialDropout1D extends Dropout {
    /** @nocollapse */
    static className: string;
    constructor(args: SpatialDropout1DLayerConfig);
    protected getNoiseShape(input: Tensor): Shape;
}
declare class Dense extends Layer {
    /** @nocollapse */
    static className: string;
    private units;
    private activation;
    private useBias;
    private kernelInitializer;
    private biasInitializer;
    private kernel;
    private bias;
    readonly DEFAULT_KERNEL_INITIALIZER: InitializerIdentifier;
    readonly DEFAULT_BIAS_INITIALIZER: InitializerIdentifier;
    private readonly kernelConstraint?;
    private readonly biasConstraint?;
    private readonly kernelRegularizer?;
    private readonly biasRegularizer?;
    constructor(args: DenseLayerArgs);
    build(inputShape: Shape | Shape[]): void;
    computeOutputShape(inputShape: Shape | Shape[]): Shape | Shape[];
    call(inputs: Tensor | Tensor[], kwargs: Kwargs): Tensor | Tensor[];
    getConfig(): serialization.ConfigDict;
}
declare interface FlattenLayerArgs extends LayerArgs {
    /** Image data format: channelsLast (default) or channelsFirst. */
    dataFormat?: DataFormat;
}
declare class Flatten extends Layer {
    private dataFormat;
    /** @nocollapse */
    static className: string;
    constructor(args?: FlattenLayerArgs);
    computeOutputShape(inputShape: Shape | Shape[]): Shape | Shape[];
    call(inputs: Tensor | Tensor[], kwargs: Kwargs): Tensor | Tensor[];
    getConfig(): serialization.ConfigDict;
}
declare interface ActivationLayerArgs extends LayerArgs {
    /**
     * Name of the activation function to use.
     */
    activation: ActivationIdentifier;
}
declare class Activation extends Layer {
    /** @nocollapse */
    static className: string;
    activation: ActivationFn;
    constructor(args: ActivationLayerArgs);
    call(inputs: Tensor | Tensor[], kwargs: Kwargs): Tensor | Tensor[];
    getConfig(): serialization.ConfigDict;
}
declare interface ReshapeLayerArgs extends LayerArgs {
    /** The target shape. Does not include the batch axis. */
    targetShape: Shape;
}
declare interface RepeatVectorLayerArgs extends LayerArgs {
    /**
     * The integer number of times to repeat the input.
     */
    n: number;
}
declare class RepeatVector extends Layer {
    /** @nocollapse */
    static className: string;
    readonly n: number;
    constructor(args: RepeatVectorLayerArgs);
    computeOutputShape(inputShape: Shape): Shape;
    call(inputs: Tensor | Tensor[], kwargs: Kwargs): Tensor | Tensor[];
    getConfig(): serialization.ConfigDict;
}
declare class Reshape extends Layer {
    /** @nocollapse */
    static className: string;
    private targetShape;
    constructor(args: ReshapeLayerArgs);
    private isUnknown;
    /**
     * Finds and replaces a missing dimension in output shape.
     *
     * This is a near direct port of the internal Numpy function
     * `_fix_unknown_dimension` in `numpy/core/src/multiarray/shape.c`.
     *
     * @param inputShape: Original shape of array begin reshape.
     * @param outputShape: Target shape of the array, with at most a single
     * `null` or negative number, which indicates an underdetermined dimension
     * that should be derived from `inputShape` and the known dimensions of
     *   `outputShape`.
     * @returns: The output shape with `null` replaced with its computed value.
     * @throws: ValueError: If `inputShape` and `outputShape` do not match.
     */
    private fixUnknownDimension;
    computeOutputShape(inputShape: Shape): Shape;
    call(inputs: Tensor | Tensor[], kwargs: Kwargs): Tensor | Tensor[];
    getConfig(): serialization.ConfigDict;
}
declare interface PermuteLayerArgs extends LayerArgs {
    /**
     * Array of integers. Permutation pattern. Does not include the
     * sample (batch) dimension. Index starts at 1.
     * For instance, `[2, 1]` permutes the first and second dimensions
     * of the input.
     */
    dims: number[];
}
declare class Permute extends Layer {
    /** @nocollapse */
    static className: string;
    readonly dims: number[];
    private readonly dimsIncludingBatch;
    constructor(args: PermuteLayerArgs);
    computeOutputShape(inputShape: Shape | Shape[]): Shape | Shape[];
    call(inputs: Tensor | Tensor[], kwargs: Kwargs): Tensor | Tensor[];
    getConfig(): serialization.ConfigDict;
}
declare interface MaskingArgs extends LayerArgs {
    /**
     * Masking Value. Defaults to `0.0`.
     */
    maskValue?: number;
}
declare class Masking extends Layer {
    /** @nocollapse */
    static className: string;
    maskValue: number;
    constructor(args?: MaskingArgs);
    computeOutputShape(inputShape: Shape | Shape[]): Shape | Shape[];
    getConfig(): {
        maskValue: number;
    };
    computeMask(inputs: Tensor | Tensor[], mask?: Tensor | Tensor[]): Tensor;
    call(inputs: Tensor | Tensor[], kwargs: Kwargs): Tensor | Tensor[];
}

/// <amd-module name="@tensorflow/tfjs-layers/dist/keras_format/layers/core_serialization" />
interface DropoutLayerConfig extends LayerConfig {
    rate: number;
    noise_shape?: number[];
    seed?: number;
}
declare type DropoutLayerSerialization = BaseLayerSerialization<'Dropout', DropoutLayerConfig>;
interface DenseLayerConfig extends LayerConfig {
    units: number;
    activation?: ActivationSerialization;
    use_bias?: boolean;
    input_dim?: number;
    kernel_initializer?: InitializerSerialization;
    bias_initializer?: InitializerSerialization;
    kernel_constraint?: ConstraintSerialization;
    bias_constraint?: ConstraintSerialization;
    kernel_regularizer?: RegularizerSerialization;
    bias_regularizer?: RegularizerSerialization;
    activity_regularizer?: RegularizerSerialization;
}
declare type DenseLayerSerialization = BaseLayerSerialization<'Dense', DenseLayerConfig>;
declare type FlattenLayerSerialization = BaseLayerSerialization<'Flatten', LayerConfig>;
interface ActivationLayerConfig extends LayerConfig {
    activation: ActivationSerialization;
}
declare type ActivationLayerSerialization = BaseLayerSerialization<'Activation', ActivationLayerConfig>;
interface RepeatVectorLayerConfig extends LayerConfig {
    n: number;
}
declare type RepeatVectorLayerSerialization = BaseLayerSerialization<'RepeatVector', RepeatVectorLayerConfig>;
interface ReshapeLayerConfig extends LayerConfig {
    target_shape: Shape;
}
declare type ReshapeLayerSerialization = BaseLayerSerialization<'Reshape', ReshapeLayerConfig>;
interface PermuteLayerConfig extends LayerConfig {
    dims: number[];
}
declare type PermuteLayerSerialization = BaseLayerSerialization<'Permute', PermuteLayerConfig>;
interface MaskingLayerConfig extends LayerConfig {
    maskValue: number;
}
declare type MaskingLayerSerialization = BaseLayerSerialization<'Masking', MaskingLayerConfig>;
declare type CoreLayerSerialization = DropoutLayerSerialization | DenseLayerSerialization | FlattenLayerSerialization | ActivationLayerSerialization | RepeatVectorLayerSerialization | ReshapeLayerSerialization | PermuteLayerSerialization | MaskingLayerSerialization;
declare type CoreLayerClassName = CoreLayerSerialization['class_name'];
/**
 * A string array of valid CoreLayer class names.
 *
 * This is guaranteed to match the `CoreLayerClassName` union type.
 */
declare const coreLayerClassNames: CoreLayerClassName[];

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/cos" />
/**
 * Computes cos of the input `tf.Tensor` element-wise: `cos(x)`
 *
 * ```js
 * const x = tf.tensor1d([0, Math.PI / 2, Math.PI * 3 / 4]);
 *
 * x.cos().print();  // or tf.cos(x)
 * ```
 * @param x The input tensor. Must be float32 type.
 *
 * @doc {heading: 'Operations', subheading: 'Basic math'}
 */
declare function cos_<T extends Tensor>(x: T | TensorLike): T;
declare const cos: typeof cos_;
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/cosh" />
/**
 * Computes hyperbolic cos of the input `tf.Tensor` element-wise: `cosh(x)`
 *
 * ```js
 * const x = tf.tensor1d([0, 1, -1, .7]);
 *
 * x.cosh().print();  // or tf.cosh(x)
 * ```
 * @param x The input tensor. Must be float32 type.
 *
 * @doc {heading: 'Operations', subheading: 'Basic math'}
 */
declare function cosh_<T extends Tensor>(x: T | TensorLike): T;
declare const cosh: typeof cosh_;
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/gradients/Cosh_grad" />
declare const coshGradConfig: GradConfig;

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/cosh_test" />
{ };
/// <amd-module name="@tensorflow/tfjs-core/dist/ops/losses/cosine_distance" />

/**
 * Computes the cosine distance loss between two tensors.
 *
 * @param labels The ground truth output tensor, same dimensions as
 *    'predictions'.
 * @param predictions The predicted outputs.
 * @param axis The dimension along which the cosine distance is computed.
 * @param weights Tensor whose rank is either 0, or the same rank as
 *    `labels`, and must be broadcastable to `labels` (i.e., all dimensions
 *    must be either `1`, or the same as the corresponding `losses`
 *    dimension).
 * @param reduction Type of reduction to apply to loss. Should be of type
 *    `Reduction`
 *
 * @doc {heading: 'Training', subheading: 'Losses', namespace: 'losses'}
 */
declare function cosineDistance_<T extends Tensor, O extends Tensor>(labels: T | TensorLike, predictions: T | TensorLike, axis: number, weights?: Tensor | TensorLike, reduction?: Reduction): O;
declare const cosineDistance: typeof cosineDistance_;
{ };
/// <amd-module name="@tensorflow/tfjs-core/dist/ops/losses/cosine_distance_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/gradients/Cos_grad" />
declare const cosGradConfig: GradConfig;

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/cos_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/image/crop_and_resize" />
/**
 * Extracts crops from the input image tensor and resizes them using bilinear
 * sampling or nearest neighbor sampling (possibly with aspect ratio change)
 * to a common output size specified by cropSize.
 *
 * @param image 4d tensor of shape `[batch,imageHeight,imageWidth, depth]`,
 *     where imageHeight and imageWidth must be positive, specifying the
 *     batch of images from which to take crops
 * @param boxes 2d float32 tensor of shape `[numBoxes, 4]`. Each entry is
 *     `[y1, x1, y2, x2]`, where `(y1, x1)` and `(y2, x2)` are the normalized
 *     coordinates of the box in the `boxInd[i]`th image in the batch
 * @param boxInd 1d int32 tensor of shape `[numBoxes]` with values in range
 *     `[0, batch)` that specifies the image that the `i`-th box refers to.
 * @param cropSize 1d int32 tensor of 2 elements `[cropHeigh, cropWidth]`
 *     specifying the size to which all crops are resized to.
 * @param method Optional string from `'bilinear' | 'nearest'`,
 *     defaults to bilinear, which specifies the sampling method for resizing
 * @param extrapolationValue A threshold for deciding when to remove boxes based
 *     on score. Defaults to 0.
 * @return A 4D tensor of the shape `[numBoxes,cropHeight,cropWidth,depth]`
 *
 * @doc {heading: 'Operations', subheading: 'Images', namespace: 'image'}
 */
declare function cropAndResize_(image: Tensor4D | TensorLike, boxes: Tensor2D | TensorLike, boxInd: Tensor1D | TensorLike, cropSize: [number, number], method?: 'bilinear' | 'nearest', extrapolationValue?: number): Tensor4D;
declare const cropAndResize: typeof cropAndResize_;
{ };
/// <amd-module name="@tensorflow/tfjs-core/dist/ops/image/crop_and_resize_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-data/dist/datasets/csv_dataset" />
/**
 * Represents a potentially large collection of delimited text records.
 *
 * The produced `TensorContainer`s each contain one key-value pair for
 * every column of the table.  When a field is empty in the incoming data, the
 * resulting value is `undefined`, or throw error if it is required.  Values
 * that can be parsed as numbers are emitted as type `number`, other values
 * are parsed as `string`.
 *
 * The results are not batched.
 *
 * @doc {heading: 'Data', subheading: 'Classes', namespace: 'data'}
 */
declare class CSVDataset extends Dataset<TensorContainer> {
    protected readonly input: DataSource;
    base: TextLineDataset;
    private hasHeader;
    private fullColumnNames;
    private columnNamesValidated;
    private columnConfigs;
    private configuredColumnsOnly;
    private delimiter;
    private delimWhitespace;
    /**
     * Returns column names of the csv dataset. If `configuredColumnsOnly` is
     * true, return column names in `columnConfigs`. If `configuredColumnsOnly` is
     * false and `columnNames` is provided, `columnNames`. If
     * `configuredColumnsOnly` is false and `columnNames` is not provided, return
     * all column names parsed from the csv file. For example usage please go to
     * `tf.data.csv`.
     *
     * @doc {heading: 'Data', subheading: 'Classes'}
     */
    columnNames(): Promise<string[]>;
    private setColumnNames;
    private maybeReadHeaderLine;
    /**
     * Create a `CSVDataset`.
     *
     * @param input A `DataSource` providing a chunked, UTF8-encoded byte stream.
     * @param csvConfig (Optional) A CSVConfig object that contains configurations
     *     of reading and decoding from CSV file(s).
     *
     *     hasHeader: (Optional) A boolean value that indicates whether the first
     *     row of provided CSV file is a header line with column names, and should
     *     not be included in the data. Defaults to `true`.
     *
     *     columnNames: (Optional) A list of strings that corresponds to
     *     the CSV column names, in order. If provided, it ignores the column
     *     names inferred from the header row. If not provided, infers the column
     *     names from the first row of the records. If hasHeader is false and
     *     columnNames is not provided, this method throws an error.
     *
     *     columnConfigs: (Optional) A dictionary whose key is column names, value
     *     is an object stating if this column is required, column's data type,
     *     default value, and if this column is label. If provided, keys must
     *     correspond to names provided in columnNames or inferred from the file
     *     header lines. If isLabel is true any column, returns an array of two
     *     items: the first item is a dict of features key/value pairs, the second
     *     item is a dict of labels key/value pairs. If no feature is marked as
     *     label, returns a dict of features only.
     *
     *     configuredColumnsOnly (Optional) If true, only columns provided in
     *     columnConfigs will be parsed and provided during iteration.
     *
     *     delimiter (Optional) The string used to parse each line of the input
     *     file. Defaults to `,`.
     */
    constructor(input: DataSource, csvConfig?: CSVConfig);
    iterator(): Promise<LazyIterator<TensorContainer>>;
    makeDataElement(line: string): TensorContainer;
    private getBoolean;
    private parseRow;
}

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/cumprod" />
/**
 * Computes the cumulative product of a `tf.Tensor` along `axis`.
 *
 * ```js
 * const x = tf.tensor([1, 2, 3, 4]);
 * x.cumprod().print();
 * ```
 * ```js
 * const x = tf.tensor([[1, 2], [3, 4]]);
 * x.cumprod().print();
 * ```
 *
 * @param x The input tensor to cumulatively multiply.
 * @param axis The axis along which to multiply. Optional. Defaults to 0.
 * @param exclusive Whether to perform exclusive cumulative product. Optional.
 *     Defaults to false. If set to true then the product of each tensor entry
 *     does not include its own value, but only the values previous to it
 *     along the specified axis.
 * @param reverse Whether to multiply in the opposite direction. Optional.
 *     Defaults to false.
 *
 * @doc {heading: 'Operations', subheading: 'Scan'}
 */
declare function cumprod_<T extends Tensor>(x: Tensor | TensorLike, axis?: number, exclusive?: boolean, reverse?: boolean): T;
declare const cumprod: typeof cumprod_;
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/cumprod_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/cumsum" />
/**
 * Computes the cumulative sum of a `tf.Tensor` along `axis`.
 *
 * ```js
 * const x = tf.tensor([1, 2, 3, 4]);
 * x.cumsum().print();
 * ```
 * ```js
 * const x = tf.tensor([[1, 2], [3, 4]]);
 * x.cumsum().print();
 * ```
 *
 * @param x The input tensor to be summed.
 * @param axis The axis along which to sum. Optional. Defaults to 0.
 * @param exclusive Whether to perform exclusive cumulative sum. Optional.
 *     Defaults to false. If set to true then the sum of each tensor entry
 *     does not include its own value, but only the values previous to it
 *     along the specified axis.
 * @param reverse Whether to sum in the opposite direction. Optional.
 *     Defaults to false.
 *
 * @doc {heading: 'Operations', subheading: 'Scan'}
 */
declare function cumsum_<T extends Tensor>(x: Tensor | TensorLike, axis?: number, exclusive?: boolean, reverse?: boolean): T;
declare const cumsum: typeof cumsum_;
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/gradients/Cumsum_grad" />
declare const cumsumGradConfig: GradConfig;

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/cumsum_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-data/dist/dataset" />
/**
 * A nested structure of Datasets, used as the input to zip().
 */
declare type DatasetContainer = Container<Dataset<TensorContainer>>;
/**
 * Represents a potentially large list of independent data elements (typically
 * 'samples' or 'examples').
 *
 * A 'data example' may be a primitive, an array, a map from string keys to
 * values, or any nested structure of these.
 *
 * A `Dataset` represents an ordered collection of elements, together with a
 * chain of transformations to be performed on those elements. Each
 * transformation is a method of `Dataset` that returns another `Dataset`, so
 * these may be chained, e.g.
 * `const processedDataset = rawDataset.filter(...).map(...).batch(...)`.
 *
 * Data loading and transformation is done in a lazy, streaming fashion.  The
 * dataset may be iterated over multiple times; each iteration starts the data
 * loading anew and recapitulates the transformations.
 *
 * A `Dataset` is typically processed as a stream of unbatched examples -- i.e.,
 * its transformations are applied one example at a time. Batching produces a
 * new `Dataset` where each element is a batch. Batching should usually come
 * last in a pipeline, because data transformations are easier to express on a
 * per-example basis than on a per-batch basis.
 *
 * The following code examples are calling `await dataset.forEachAsync(...)` to
 * iterate once over the entire dataset in order to print out the data.
 *
 * @doc {heading: 'Data', subheading: 'Classes', namespace: 'data'}
 */
declare abstract class Dataset<T extends tf.TensorContainer> {
    abstract iterator(): Promise<LazyIterator<T>>;
    readonly size: number;
    /**
     * Groups elements into batches.
     *
     * It is assumed that each of the incoming dataset elements has the same
     * structure -- i.e. the same set of keys at each location in an object
     * hierarchy.  For each key, the resulting `Dataset` provides a batched
     * element collecting all of the incoming values for that key.
     *
     *  * Incoming primitives are grouped into a 1-D Tensor.
     *  * Incoming Tensors are grouped into a new Tensor where the 0th axis is
     *    the batch dimension.
     *  * Incoming arrays are converted to Tensor and then batched.
     *  * A nested array is interpreted as an n-D Tensor, so the batched result
     *    has n+1 dimensions.
     *  * An array that cannot be converted to Tensor produces an error.
     *
     * If an array should not be batched as a unit, it should first be converted
     * to an object with integer keys.
     *
     * Here are a few examples:
     *
     * Batch a dataset of numbers:
     * ```js
     * const a = tf.data.array([1, 2, 3, 4, 5, 6, 7, 8]).batch(4);
     * await a.forEachAsync(e => e.print());
     * ```
     *
     * Batch a dataset of arrays:
     * ```js
     * const b = tf.data.array([[1], [2], [3], [4], [5], [6], [7], [8]]).batch(4);
     * await b.forEachAsync(e => e.print());
     * ```
     *
     * Batch a dataset of objects:
     * ```js
     * const c = tf.data.array([{a: 1, b: 11}, {a: 2, b: 12}, {a: 3, b: 13},
     *   {a: 4, b: 14}, {a: 5, b: 15}, {a: 6, b: 16}, {a: 7, b: 17},
     *   {a: 8, b: 18}]).batch(4);
     * await c.forEachAsync(e => {
     *   console.log('{');
     *   for(var key in e) {
     *     console.log(key+':');
     *     e[key].print();
     *   }
     *   console.log('}');
     * })
     * ```
     *
     * @param batchSize The number of elements desired per batch.
     * @param smallLastBatch Whether to emit the final batch when it has fewer
     *   than batchSize elements. Default true.
     * @returns A `Dataset`, from which a stream of batches can be obtained.
     *
     * @doc {heading: 'Data', subheading: 'Classes'}
     */
    batch(batchSize: number, smallLastBatch?: boolean): Dataset<tf.TensorContainer>;
    /**
     * Concatenates this `Dataset` with another.
     *
     * ```js
     * const a = tf.data.array([1, 2, 3]);
     * const b = tf.data.array([4, 5, 6]);
     * const c = a.concatenate(b);
     * await c.forEachAsync(e => console.log(e));
     * ```
     *
     * @param dataset A `Dataset` to be concatenated onto this one.
     * @returns A `Dataset`.
     *
     * @doc {heading: 'Data', subheading: 'Classes'}
     */
    concatenate(dataset: Dataset<T>): Dataset<T>;
    /**
     * Filters this dataset according to `predicate`.
     *
     * ```js
     * const a = tf.data.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
     *   .filter(x => x%2 === 0);
     * await a.forEachAsync(e => console.log(e));
     * ```
     *
     * @param predicate A function mapping a dataset element to a boolean or a
     * `Promise` for one.
     *
     * @returns A `Dataset` of elements for which the predicate was true.
     *
     * @doc {heading: 'Data', subheading: 'Classes'}
     */
    filter(predicate: (value: T) => boolean): Dataset<T>;
    /**
     * Apply a function to every element of the dataset.
     *
     * After the function is applied to a dataset element, any Tensors contained
     * within that element are disposed.
     *
     * ```js
     * const a = tf.data.array([1, 2, 3]);
     * await a.forEachAsync(e => console.log(e));
     * ```
     *
     * @param f A function to apply to each dataset element.
     * @returns A `Promise` that resolves after all elements have been processed.
     *
     * @doc {heading: 'Data', subheading: 'Classes'}
     */
    forEachAsync(f: (input: T) => void): Promise<void>;
    /**
     * Maps this dataset through a 1-to-1 transform.
     *
     * ```js
     * const a = tf.data.array([1, 2, 3]).map(x => x*x);
     * await a.forEachAsync(e => console.log(e));
     * ```
     *
     * @param transform A function mapping a dataset element to a transformed
     *   dataset element.
     *
     * @returns A `Dataset` of transformed elements.
     *
     * @doc {heading: 'Data', subheading: 'Classes'}
     */
    map<O extends tf.TensorContainer>(transform: (value: T) => O): Dataset<O>;
    /**
     * Maps this dataset through an async 1-to-1 transform.
     *
     * ```js
     * const a =
     *  tf.data.array([1, 2, 3]).mapAsync(x => new Promise(function(resolve){
     *    setTimeout(() => {
     *      resolve(x * x);
     *    }, Math.random()*1000 + 500);
     *  }));
     * console.log(await a.toArray());
     * ```
     *
     * @param transform A function mapping a dataset element to a `Promise` for a
     *   transformed dataset element.  This transform is responsible for disposing
     *   any intermediate `Tensor`s, i.e. by wrapping its computation in
     *   `tf.tidy()`; that cannot be automated here (as it is in the synchronous
     *   `map()` case).
     *
     * @returns A `Dataset` of transformed elements.
     *
     * @doc {heading: 'Data', subheading: 'Classes'}
     */
    mapAsync<O extends tf.TensorContainer>(transform: (value: T) => Promise<O>): Dataset<O>;
    /**
     *  Creates a `Dataset` that prefetches elements from this dataset.
     *
     * @param bufferSize: An integer specifying the number of elements to be
     *   prefetched.
     * @returns A `Dataset`.
     *
     * @doc {heading: 'Data', subheading: 'Classes'}
     */
    prefetch(bufferSize: number): Dataset<T>;
    /**
     * Repeats this dataset `count` times.
     *
     * NOTE: If this dataset is a function of global state (e.g. a random number
     * generator), then different repetitions may produce different elements.
     *
     * ```js
     * const a = tf.data.array([1, 2, 3]).repeat(3);
     * await a.forEachAsync(e => console.log(e));
     * ```
     *
     * @param count: (Optional) An integer, representing the number of times
     *   the dataset should be repeated. The default behavior (if `count` is
     *   `undefined` or negative) is for the dataset be repeated indefinitely.
     * @returns A `Dataset`.
     *
     * @doc {heading: 'Data', subheading: 'Classes'}
     */
    repeat(count?: number): Dataset<T>;
    /**
     * Creates a `Dataset` that skips `count` initial elements from this dataset.
     *
     * ```js
     * const a = tf.data.array([1, 2, 3, 4, 5, 6]).skip(3);
     * await a.forEachAsync(e => console.log(e));
     * ```
     *
     * @param count: The number of elements of this dataset that should be skipped
     *   to form the new dataset.  If `count` is greater than the size of this
     *   dataset, the new dataset will contain no elements.  If `count`
     *   is `undefined` or negative, skips the entire dataset.
     *
     * @returns A `Dataset`.
     *
     * @doc {heading: 'Data', subheading: 'Classes'}
     */
    skip(count: number): Dataset<T>;
    static readonly MAX_BUFFER_SIZE = 10000;
    /**
     * Pseudorandomly shuffles the elements of this dataset. This is done in a
     * streaming manner, by sampling from a given number of prefetched elements.
     *
     * ```js
     * const a = tf.data.array([1, 2, 3, 4, 5, 6]).shuffle(3);
     * await a.forEachAsync(e => console.log(e));
     * ```
     *
     * @param bufferSize: An integer specifying the number of elements from this
     *   dataset from which the new dataset will sample.
     * @param seed: (Optional) An integer specifying the random seed that will
     *   be used to create the distribution.
     * @param reshuffleEachIteration: (Optional) A boolean, which if true
     *   indicates that the dataset should be pseudorandomly reshuffled each time
     *   it is iterated over. If false, elements will be returned in the same
     *   shuffled order on each iteration. (Defaults to `true`.)
     * @returns A `Dataset`.
     *
     * @doc {heading: 'Data', subheading: 'Classes'}
     */
    shuffle(bufferSize: number, seed?: string, reshuffleEachIteration?: boolean): Dataset<T>;
    /**
     * Creates a `Dataset` with at most `count` initial elements from this
     * dataset.
     *
     * ```js
     * const a = tf.data.array([1, 2, 3, 4, 5, 6]).take(3);
     * await a.forEachAsync(e => console.log(e));
     * ```
     *
     * @param count: The number of elements of this dataset that should be taken
     *   to form the new dataset.  If `count` is `undefined` or negative, or if
     *   `count` is greater than the size of this dataset, the new dataset will
     *   contain all elements of this dataset.
     * @returns A `Dataset`.
     *
     * @doc {heading: 'Data', subheading: 'Classes'}
     */
    take(count: number): Dataset<T>;
    /**
     * Collect all elements of this dataset into an array.
     *
     * Obviously this will succeed only for small datasets that fit in memory.
     * Useful for testing and generally should be avoided if possible.
     *
     * ```js
     * const a = tf.data.array([1, 2, 3, 4, 5, 6]);
     * console.log(await a.toArray());
     * ```
     *
     * @returns A Promise for an array of elements, which will resolve
     *   when a new stream has been obtained and fully consumed.
     *
     * @doc {heading: 'Data', subheading: 'Classes'}
     */
    toArray(): Promise<T[]>;
    /**
     * Collect all elements of this dataset into an array with prefetching 100
     * elements. This is useful for testing, because the prefetch changes the
     * order in which the Promises are resolved along the processing pipeline.
     * This may help expose bugs where results are dependent on the order of
     * Promise resolution rather than on the logical order of the stream (i.e.,
     * due to hidden mutable state).
     *
     * @returns A Promise for an array of elements, which will resolve
     *   when a new stream has been obtained and fully consumed.
     */
    toArrayForTest(): Promise<T[]>;
}
/**
 * Create a `Dataset` defined by a provided iterator() function.
 *
 * ```js
 * let i = -1;
 * const func = () =>
 *    ++i < 5 ? {value: i, done: false} : {value: null, done: true};
 * const iter = tf.data.iteratorFromFunction(func);
 * const ds = tf.data.datasetFromIteratorFn(iter);
 * await ds.forEachAsync(e => console.log(e));
 * ```
 */
declare function datasetFromIteratorFn<T extends tf.TensorContainer>(iteratorFn: () => Promise<LazyIterator<T>>, size?: number): Dataset<T>;
/**
 * Create a `Dataset` from an array of elements.
 *
 * Create a Dataset from an array of objects:
 * ```js
 * const a = tf.data.array([{'item': 1}, {'item': 2}, {'item': 3}]);
 * await a.forEachAsync(e => console.log(e));
 * ```
 *
 * Create a Dataset from an array of numbers:
 * ```js
 * const a = tf.data.array([4, 5, 6]);
 * await a.forEachAsync(e => console.log(e));
 * ```
 * @param items An array of elements that will be parsed as items in a dataset.
 *
 * @doc {heading: 'Data', subheading: 'Creation', namespace: 'data'}
 */
declare function array<T extends tf.TensorContainer>(items: T[]): Dataset<T>;
/**
 * Create a `Dataset` by zipping together an array, dict, or nested
 * structure of `Dataset`s (and perhaps additional constants).
 * The underlying datasets must provide elements in a consistent order such that
 * they correspond.
 *
 * The number of elements in the resulting dataset is the same as the size of
 * the smallest dataset in datasets.
 *
 * The nested structure of the `datasets` argument determines the
 * structure of elements in the resulting iterator.
 *
 * Note this means that, given an array of two datasets that produce dict
 * elements, the result is a dataset that produces elements that are arrays
 * of two dicts:
 *
 * Zip an array of datasets:
 * ```js
 * console.log('Zip two datasets of objects:');
 * const ds1 = tf.data.array([{a: 1}, {a: 2}, {a: 3}]);
 * const ds2 = tf.data.array([{b: 4}, {b: 5}, {b: 6}]);
 * const ds3 = tf.data.zip([ds1, ds2]);
 * await ds3.forEachAsync(e => console.log(JSON.stringify(e)));
 *
 * // If the goal is to merge the dicts in order to produce elements like
 * // {a: ..., b: ...}, this requires a second step such as:
 * console.log('Merge the objects:');
 * const ds4 = ds3.map(x => {return {a: x[0].a, b: x[1].b}});
 * await ds4.forEachAsync(e => console.log(e));
 * ```
 *
 * Zip a dict of datasets:
 * ```js
 * const a = tf.data.array([{a: 1}, {a: 2}, {a: 3}]);
 * const b = tf.data.array([{b: 4}, {b: 5}, {b: 6}]);
 * const c = tf.data.zip({c: a, d: b});
 * await c.forEachAsync(e => console.log(JSON.stringify(e)));
 * ```
 *
 * @doc {heading: 'Data', subheading: 'Operations', namespace: 'data'}
 */
declare function zip<O extends tf.TensorContainer>(datasets: DatasetContainer): Dataset<O>;

/// <amd-module name="@tensorflow/tfjs-layers/dist/engine/dataset_fakes" />
interface FakeDatasetArgs {
    /**
     * The shape(s) of the features of a single example.
     *
     * Use an object mapping name to shape, if more than one feature tensors
     * are required.
     */
    xShape: Shape | {
        [name: string]: Shape;
    };
    /**
     * The shape of the target(s) of a single exapmle.
     */
    yShape: Shape | {
        [name: string]: Shape;
    };
    /**
     * A function that generates preset sequence of X tensors.
     *
     * This function is invoked each time a new iterator is created.
     */
    xTensorsFunc?: () => tfc.Tensor[] | {
        [name: string]: tfc.Tensor[];
    };
    /**
     * A function that generates preset sequence of Y tensors.
     *
     * This function is invoked each time a new iterator is created.
     */
    yTensorsFunc?: () => tfc.Tensor[] | {
        [name: string]: tfc.Tensor[];
    };
    /**
     * The size of each batch generated by the iterator.
     */
    batchSize: number;
    /**
     * The number of batches an iterator generates before declaring done to be
     * true.
     */
    numBatches: number;
}
/**
 * A fake dataset with configurable feature and target shapes.
 *
 * The batch size and # of batches are also configurable.
 *
 * The iterator from the dataset always generate random-normal float32 values.
 */
declare class FakeNumericDataset extends Dataset<FitDatasetElement> {
    readonly args: FakeDatasetArgs;
    constructor(args: FakeDatasetArgs);
    iterator(): Promise<LazyIterator<FitDatasetElement>>;
}
declare class FakeNumericDatasetLegacyArrayForm extends Dataset<[TensorOrArrayOrMap, TensorOrArrayOrMap]> {
    readonly args: FakeDatasetArgs;
    ds: FakeNumericDataset;
    constructor(args: FakeDatasetArgs);
    iterator(): Promise<LazyIterator<[TensorOrArrayOrMap, TensorOrArrayOrMap]>>;
}

/// <amd-module name="@tensorflow/tfjs-layers/dist/engine/dataset_stub" />
/**
 * Stub interfaces and classes for testing tf.LayersModel.fitDataset().
 *
 * TODO(cais, soergel): Remove this in favor of actual interfaces and classes
 *   when ready.
 */
declare abstract class LazyIterator<T> {
    abstract next(): Promise<IteratorResult<T>>;
}
declare abstract class Dataset<T> {
    abstract iterator(): Promise<LazyIterator<T>>;
    size: number;
}

/// <amd-module name="@tensorflow/tfjs-data/dist/datasource" />
/**
 * Represents a data source readable as a stream of binary data chunks.
 *
 * Because `Dataset`s can be read repeatedly (via `Dataset.iterator()`), this
 * provides a means to repeatedly create streams from the underlying data
 * sources.
 */
declare abstract class DataSource {
    /**
     * Obtain a new stream of binary data chunks.
     *
     * Starts the new stream from the beginning of the data source, even if other
     * streams have been obtained previously.
     */
    abstract iterator(): Promise<ByteChunkIterator>;
}

/// <amd-module name="@tensorflow/tfjs-core/dist/debug_mode_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-data/dist/util/deep_clone" />
declare function deepClone<T>(container: T): T;

/// <amd-module name="@tensorflow/tfjs-data/dist/util/deep_map" />
/**
 * A return value for a mapping function that can be applied via deepMap.
 *
 * If recurse is true, the value should be empty, and iteration will continue
 * into the object or array.
 */
declare type DeepMapResult = {
    value: any;
    recurse: boolean;
};
/**
 * Apply a mapping function to a nested structure in a recursive manner.
 *
 * The result of the mapping is an object with the same nested structure (i.e.,
 * of arrays and dicts) as the input, except that some subtrees are replaced,
 * according to the results of the mapping function.
 *
 * Mappings are memoized.  Thus, if the nested structure contains the same
 * object in multiple positions, the output will contain the same mapped object
 * in those positions.  Cycles are not supported, however.
 *
 * @param input: The object to which to apply the mapping function.
 * @param mapFn: A function that expects a single node of the object tree, and
 *   returns a `DeepMapResult`.  The `DeepMapResult` either provides a
 *   replacement value for that node (i.e., replacing the subtree), or indicates
 *   that the node should be processed recursively.
 */
declare function deepMap(input: any, mapFn: (x: any) => DeepMapResult): any | any[];
/**
 * Zip nested structures together in a recursive manner.
 *
 * This has the effect of transposing or pivoting data, e.g. converting it from
 * a row-major representation to a column-major representation.
 *
 * For example, `deepZip([{a: 1, b: 2}, {a: 3, b: 4}])` returns
 * `{a: [1, 3], b: [2, 4]}`.
 *
 * The inputs should all have the same nested structure (i.e., of arrays and
 * dicts).  The result is a single object with the same nested structure, where
 * the leaves are arrays collecting the values of the inputs at that location
 * (or, optionally, the result of a custom function applied to those arrays).
 *
 * @param inputs: An array of the objects to zip together.
 * @param zipFn: (optional) A function that expects an array of elements at a
 *   single node of the object tree, and returns a `DeepMapResult`.  The
 *   `DeepMapResult` either provides a result value for that node (i.e.,
 *   representing the subtree), or indicates that the node should be processed
 *   recursively.  The default zipFn recurses as far as possible and places
 *   arrays at the leaves.
 */
declare function deepZip(inputs: any[], zipFn?: (xs: any[]) => DeepMapResult): any | any[];
declare function zipToList(x: any[]): DeepMapResult;
/**
 * A return value for an async map function for use with deepMapAndAwaitAll.
 *
 * If recurse is true, the value should be empty, and iteration will continue
 * into the object or array.
 */
declare type DeepMapAsyncResult = {
    value: Promise<any>;
    recurse: boolean;
};
/**
 * Apply an async mapping function to a nested structure in a recursive manner.
 *
 * This first creates a nested structure of Promises, and then awaits all of
 * those, resulting in a single Promise for a resolved nested structure.
 *
 * The result of the mapping is an object with the same nested structure (i.e.,
 * of arrays and dicts) as the input, except that some subtrees are replaced,
 * according to the results of the mapping function.
 *
 * Mappings are memoized.  Thus, if the nested structure contains the same
 * object in multiple positions, the output will contain the same mapped object
 * in those positions.  Cycles are not supported, however.
 *
 * @param input: The object to which to apply the mapping function.
 * @param mapFn: A function that expects a single node of the object tree, and
 *   returns a `DeepMapAsyncResult`.  The `DeepMapAsyncResult` either provides
 *   a `Promise` for a replacement value for that node (i.e., replacing the
 *   subtree), or indicates that the node should be processed recursively.  Note
 *   that the decision whether or not to recurse must be made immediately; only
 *   the mapped value may be promised.
 */
declare function deepMapAndAwaitAll(input: any, mapFn: (x: any) => DeepMapAsyncResult): Promise<any | any[]>;
/**
 * Determine whether the argument is iterable.
 *
 * @returns true if the argument is an array or any non-Tensor object.
 */
declare function isIterable(obj: any): boolean;
/**
 * Determine whether the argument can be converted to Tensor.
 *
 * Tensors, primitives, arrays, and TypedArrays all qualify; anything else does
 * not.
 *
 * @returns true if the argument can be converted to Tensor.
 */
declare function canTensorify(obj: any): boolean;

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/dense_bincount" />
/**
 * Outputs a vector with length `size` and the same dtype as `weights`.
 *
 * If `weights` are empty, then index `i` stores the number of times the value
 * `i` is counted in `x`. If `weights` are non-empty, then index `i` stores the
 * sum of the value in `weights` at each index where the corresponding value in
 * `x` is `i`.
 *
 * Values in `x` outside of the range [0, size) are ignored.
 *
 * @param x The input int tensor, rank 1 or rank 2.
 * @param weights The weights tensor, must have the same shape as x, or a
 *     length-0 Tensor, in which case it acts as all weights equal to 1.
 * @param size Non-negative integer.
 * @param binaryOutput Optional. Whether the kernel should count the appearance
 *     or number of occurrences. Defaults to False.
 *
 * @doc {heading: 'Operations', subheading: 'Reduction'}
 */
declare function denseBincount_<T extends Tensor1D | Tensor2D>(x: T | TensorLike, weights: T | TensorLike, size: number, binaryOutput?: boolean): T;
declare const denseBincount: typeof denseBincount_;
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/dense_bincount_test" />
{ };
/// <amd-module name="@tensorflow/tfjs-core/dist/gradients/DepthwiseConv2dNative_grad" />
declare const depthwiseConv2dNativeGradConfig: GradConfig;
/// <amd-module name="@tensorflow/tfjs-core/dist/public/chained_ops/depthwise_conv2d" />
declare module '../../tensor' {
    interface Tensor<R extends Rank = Rank> {
        depthwiseConv2d<T extends Tensor3D | Tensor4D>(filter: Tensor4D | TensorLike4D, strides: [number, number] | number, pad: 'valid' | 'same' | number, dataFormat?: 'NHWC' | 'NCHW', dilations?: [number, number] | number, dimRoundingMode?: 'floor' | 'round' | 'ceil'): T;
    }
}
/// <amd-module name="@tensorflow/tfjs-core/dist/ops/depthwise_conv2d_native_backprop_filter" />
declare function depthwiseConv2dNativeBackpropFilter_<T extends Tensor3D | Tensor4D>(x: T, dy: T, filterShape: [number, number, number, number], strides: [number, number] | number, pad: 'valid' | 'same' | number | ExplicitPadding, dilations?: [number, number] | number, dimRoundingMode?: 'floor' | 'round' | 'ceil'): Tensor4D;
declare const depthwiseConv2dNativeBackpropFilter: typeof depthwiseConv2dNativeBackpropFilter_;
{ };
/// <amd-module name="@tensorflow/tfjs-core/dist/ops/depthwise_conv2d_native_backprop_input" />
declare function depthwiseConv2dNativeBackpropInput_<T extends Tensor3D | Tensor4D>(xShape: [number, number, number, number], dy: T, filter: Tensor4D, strides: [number, number] | number, pad: 'valid' | 'same' | number | ExplicitPadding, dilations?: [number, number] | number, dimRoundingMode?: 'floor' | 'round' | 'ceil'): T;
declare const depthwiseConv2dNativeBackpropInput: typeof depthwiseConv2dNativeBackpropInput_;
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/depthwise_conv2d_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/depth_to_space" />
/**
 * Rearranges data from depth into blocks of spatial data. More specifically,
 * this op outputs a copy of the input tensor where values from the `depth`
 * dimension are moved in spatial blocks to the `height` and `width` dimensions.
 * The attr `blockSize` indicates the input block size and how the data is
 * moved.
 *
 *  - Chunks of data of size `blockSize * blockSize` from depth are rearranged
 * into non-overlapping blocks of size `blockSize x blockSize`
 *
 *  - The width the output tensor is `inputWidth * blockSize`, whereas the
 * height is `inputHeight * blockSize`
 *
 *  - The Y, X coordinates within each block of the output image are determined
 * by the high order component of the input channel index
 *
 *  - The depth of the input tensor must be divisible by `blockSize *
 * blockSize`
 *
 * The `dataFormat` attr specifies the layout of the input and output tensors
 * with the following options: "NHWC": [ `batch, height, width, channels` ]
 * "NCHW": [ `batch, channels, height, width` ]
 *
 * ```js
 * const x = tf.tensor4d([1, 2, 3, 4], [1, 1, 1, 4]);
 * const blockSize = 2;
 * const dataFormat = "NHWC";
 *
 * tf.depthToSpace(x, blockSize, dataFormat).print();
 * ```
 *
 * @param x The input tensor of rank 4
 * @param blockSIze  An `int` that is `>= 2`. The size of the spatial block
 * @param dataFormat An optional string from: "NHWC", "NCHW". Defaults to "NHWC"
 *
 * @doc {heading: 'Tensors', subheading: 'Transformations'}
 */
declare function depthToSpace_(x: Tensor4D | TensorLike4D, blockSize: number, dataFormat?: 'NHWC' | 'NCHW'): Tensor4D;
declare const depthToSpace: typeof depthToSpace_;
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/depth_to_space_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/device_util" />
declare function mockIsMobile(value: boolean | undefined): void;
declare function isMobile(nav?: Navigator): boolean;
declare function isBrowser(): boolean;

/// <amd-module name="@tensorflow/tfjs-core/dist/device_util_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/diag" />
/**
 * Returns a diagonal tensor with given diagonal values.
 *
 * Given a diagonal, this operation returns a tensor with the diagonal and
 * everything else padded with zeros.
 *
 * Assume the input has dimensions `[D1,..., Dk]`, then the output is a tensor
 * of rank 2k with dimensions `[D1,..., Dk, D1,..., Dk]`
 *
 * ```js
 * const x = tf.tensor1d([1, 2, 3, 4]);
 *
 * tf.diag(x).print()
 * ```
 * ```js
 * const x = tf.tensor2d([1, 2, 3, 4, 5, 6, 7, 8], [4, 2])
 *
 * tf.diag(x).print()
 * ```
 * @param x The input tensor.
 *
 * @doc {heading: 'Tensors', subheading: 'Creation'}
 */
declare function diag_(x: Tensor): Tensor;
declare const diag: typeof diag_;
{ };
/// <amd-module name="@tensorflow/tfjs-core/dist/ops/diag_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/dilation2d" />
/**
 * Computes the grayscale dilation over the input `x`.
 *
 * @param x The input tensor, rank 3 or rank 4 of shape
 *     `[batch, height, width, depth]`. If rank 3, batch of 1 is assumed.
 * @param filter The filter tensor, rank 3, of shape
 *     `[filterHeight, filterWidth, depth]`.
 * @param strides The strides of the sliding window for each dimension of the
 *     input tensor: `[strideHeight, strideWidth]`.
 *     If `strides` is a single number,
 *     then `strideHeight == strideWidth`.
 * @param pad The type of padding algorithm.
 *    - `same` and stride 1: output will be of same size as input,
 *       regardless of filter size.
 *    - `valid`: output will be smaller than input if filter is larger
 *       than 1*1x1.
 *    - For more info, see this guide:
 *     [https://www.tensorflow.org/api_docs/python/tf/nn/convolution](
 *          https://www.tensorflow.org/api_docs/python/tf/nn/convolution)
 * @param dataFormat Specify the data format of the input and output data.
 *      Defaults to 'NHWC'. Only 'NHWC' is currently supported. With the
 *      default format "NHWC", the data is stored in the order of: [batch,
 *      height, width, channels].
 * @param dilations The dilation rates: `[dilationHeight, dilationWidth]`
 *     in which we sample input values across the height and width dimensions
 *     for atrous morphological dilation. Defaults to `[1, 1]`. If `dilations`
 *     is a single number, then `dilationHeight == dilationWidth`. If it is
 *     greater than 1, then all values of `strides` must be 1.
 *
 * @doc {heading: 'Operations', subheading: 'Convolution'}
 */
declare function dilation2d_<T extends Tensor3D | Tensor4D>(x: T | TensorLike, filter: Tensor3D | TensorLike, strides: [number, number] | number, pad: 'valid' | 'same', dilations?: [number, number] | number, dataFormat?: 'NHWC'): T;
declare const dilation2d: typeof dilation2d_;
{ };
/// <amd-module name="@tensorflow/tfjs-core/dist/gradients/Dilation2D_grad" />
declare const dilation2dGradConfig: GradConfig;

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/dilation2d_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/div" />
/**
 * Divides two `tf.Tensor`s element-wise, A / B. Supports broadcasting.
 *
 * ```js
 * const a = tf.tensor1d([1, 4, 9, 16]);
 * const b = tf.tensor1d([1, 2, 3, 4]);
 *
 * a.div(b).print();  // or tf.div(a, b)
 * ```
 *
 * ```js
 * // Broadcast div a with b.
 * const a = tf.tensor1d([2, 4, 6, 8]);
 * const b = tf.scalar(2);
 *
 * a.div(b).print();  // or tf.div(a, b)
 * ```
 *
 * @param a The first tensor as the numerator.
 * @param b The second tensor as the denominator. Must have the same dtype as
 * `a`.
 *
 * @doc {heading: 'Operations', subheading: 'Arithmetic'}
 */
declare function div_<T extends Tensor>(a: Tensor | TensorLike, b: Tensor | TensorLike): T;
declare const div: typeof div_;
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/public/chained_ops/div_no_nan" />
declare module '../../tensor' {
    interface Tensor<R extends Rank = Rank> {
        divNoNan<T extends Tensor>(b: Tensor | TensorLike): T;
    }
}

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/dot" />
/**
 * Computes the dot product of two matrices and/or vectors, `t1` and `t2`.
 *
 * ```js
 * const a = tf.tensor1d([1, 2]);
 * const b = tf.tensor2d([[1, 2], [3, 4]]);
 * const c = tf.tensor2d([[1, 2, 3], [4, 5, 6]]);
 *
 * a.dot(b).print();  // or tf.dot(a, b)
 * b.dot(a).print();
 * b.dot(c).print();
 * ```
 * @param t1 The first tensor in the dot operation.
 * @param t2 The second tensor in the dot operation.
 *
 * @doc {heading: 'Operations', subheading: 'Matrices'}
 */
declare function dot_(t1: Tensor | TensorLike, t2: Tensor | TensorLike): Tensor;
declare const dot: typeof dot_;
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/dropout" />
/**
 * Computes dropout.
 *
 * ```js
 * const x = tf.tensor1d([1, 2, 2, 1]);
 * const rate = 0.75;
 * const output = tf.dropout(x, rate);
 * output.print();
 * ```
 *
 * @param x A floating point Tensor or TensorLike.
 * @param rate A float in the range [0, 1). The probability that each element
 *   of x is discarded.
 * @param noiseShape An array of numbers of type int32, representing the
 * shape for randomly generated keep/drop flags. If the noiseShape has null
 * value, it will be automatically replaced with the x's relative dimension
 * size. Optional.
 * @param seed Used to create random seeds. Optional.
 * @returns A Tensor of the same shape of x.
 *
 * @doc {heading: 'Operations', subheading: 'Dropout'}
 */
declare function dropout_(x: Tensor | TensorLike, rate: number, noiseShape?: number[], seed?: number | string): Tensor;
declare const dropout: typeof dropout_;
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/dropout_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/dropout_util" />
/**
 * Normalize noise shape based on provided tensor and noise shape.
 *
 * @param x Tensor.
 * @param noiseShape The shape for the randomly generated keep/drop flags, as
 *   an array of numbers. Optional.
 * @returns Normalized noise shape.
 */
declare function getNoiseShape(x: Tensor, noiseShape?: number[]): number[];

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/dropout_util_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/einsum" />
/**
 * Tensor contraction over specified indices and outer product.
 *
 * `einsum` allows defining Tensors by defining their element-wise computation.
 * This computation is based on
 * [Einstein summation](https://en.wikipedia.org/wiki/Einstein_notation).
 *
 * Some special cases include:
 *
 * Matrix multiplication:
 * ```js
 * const x = tf.tensor2d([[1, 2, 3], [4, 5, 6]]);
 * const y = tf.tensor2d([[0, 1], [2, 3], [4, 5]]);
 * x.print();
 * y.print();
 * tf.einsum('ij,jk->ik', x, y).print();
 * ```
 *
 * Dot product:
 * ```js
 * const x = tf.tensor1d([1, 2, 3]);
 * const y = tf.tensor1d([0, 1, 2]);
 * x.print();
 * y.print();
 * tf.einsum('i,i->', x, y).print();
 * ```
 *
 * Batch dot product:
 * ```js
 * const x = tf.tensor2d([[1, 2, 3], [4, 5, 6]]);
 * const y = tf.tensor2d([[0, 1, 2], [3, 4, 5]]);
 * x.print();
 * y.print();
 * tf.einsum('bi,bi->b', x, y).print();
 * ```
 *
 * Outer prouduct:
 * ```js
 * const x = tf.tensor1d([1, 3, 5]);
 * const y = tf.tensor1d([2, 4, 6]);
 * x.print();
 * y.print();
 * tf.einsum('i,j->ij', x, y).print();
 * ```
 *
 * Matrix transpose:
 * ```js
 * const x = tf.tensor2d([[1, 2], [3, 4]]);
 * x.print();
 * tf.einsum('ij->ji', x).print();
 * ```
 *
 * Batch matrix transpose:
 * ```js
 * const x = tf.tensor3d([[[1, 2], [3, 4]], [[-1, -2], [-3, -4]]]);
 * x.print();
 * tf.einsum('bij->bji', x).print();
 * ```
 *
 * Limitations:
 *
 * This implementation of einsum has the following limitations:
 *
 * - Does not support >2 input tensors.
 * - Does not support duplicate axes for any given input tensor. E.g., equation
 *   'ii->' is not supported.
 * - The `...` notation is not supported.
 *
 * @param equation a string describing the contraction, in the same format as
 * [numpy.einsum](https://numpy.org/doc/stable/reference/generated/numpy.einsum.html).
 * @param tensors the input(s) to contract (each one a Tensor), whose shapes
 *     should be consistent with equation.
 * @returns The output tensor.
 *
 * @doc {heading: 'Tensors', subheading: 'Matrices'}
 */
declare function einsum_(equation: string, ...tensors: Tensor[]): Tensor;
declare const einsum: typeof einsum_;

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/einsum_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/backends/einsum_util" />
/**
 * Utility functions for computing einsum (tensor contraction and summation
 * based on Einstein summation.)
 */
/**
 * Parse an equation for einsum.
 *
 * @param equation The einsum equation (e.g., "ij,jk->ik").
 * @param numTensors Number of tensors provided along with `equation`. Used to
 *   check matching number of input tensors.
 * @returns An object consisting of the following fields:
 *   - allDims: all dimension names as strings.
 *   - summedDims: a list of all dimensions being summed over, as indices to
 *     the elements of `allDims`.
 *   - idDims: indices of the dimensions in each input tensor, as indices to
 *     the elements of `allDims.
 */
declare function decodeEinsumEquation(equation: string, numTensors: number): {
    allDims: string[];
    summedDims: number[];
    idDims: number[][];
};
/**
 * Get the permutation for a given input tensor.
 *
 * @param nDims Total number of dimension of all tensors involved in the einsum
 *   operation.
 * @param idDims Dimension indices involve in the tensor in question.
 * @returns An object consisting of the following fields:
 *   - permutationIndices: Indices to permute the axes of the tensor with.
 *   - expandDims: Indices to the dimension that need to be expanded from the
 *     tensor after permutation.
 */
declare function getEinsumPermutation(nDims: number, idDims: number[]): {
    permutationIndices: number[];
    expandDims: number[];
};
/**
 * Checks that the dimension sizes from different input tensors match the
 * equation.
 */
declare function checkEinsumDimSizes(nDims: number, idDims: number[][], tensors: Tensor[]): void;
/**
 * Gets path of computation for einsum.
 *
 * @param summedDims indices to the dimensions being summed over.
 * @param idDims A look up table for the dimensions present in each input
 *     tensor. Each consituent array contains indices for the dimensions in the
 *     corresponding input tensor.
 *
 * @return A map with two fields:
 *   - path: The path of computation, with each element indicating the dimension
 *     being summed over after the element-wise multiplication in that step.
 *   - steps: With the same length as `path`. Each element contains the indices
 *     to the input tensors being used for element-wise multiplication in the
 *     corresponding step.
 */
declare function getEinsumComputePath(summedDims: number[], idDims: number[][]): {
    path: number[];
    steps: number[][];
};
/** Determines if an axes permutation is the identity permutation. */
declare function isIdentityPermutation(perm: number[]): boolean;
/// <amd-module name="@tensorflow/tfjs-core/dist/public/chained_ops/elu" />
declare module '../../tensor' {
    interface Tensor<R extends Rank = Rank> {
        elu<T extends Tensor>(): T;
    }
}
/// <amd-module name="@tensorflow/tfjs-core/dist/gradients/Elu_grad" />
declare const eluGradConfig: GradConfig;

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/elu_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-layers/dist/layers/embeddings" />
/**
 * TensorFlow.js Layers: Embedding Layer.
 *
 * Original source: keras/constraints.py
 */
declare interface EmbeddingLayerArgs extends LayerArgs {
    /**
     * Integer > 0. Size of the vocabulary, i.e. maximum integer index + 1.
     */
    inputDim: number;
    /**
     * Integer >= 0. Dimension of the dense embedding.
     */
    outputDim: number;
    /**
     * Initializer for the `embeddings` matrix.
     */
    embeddingsInitializer?: InitializerIdentifier | Initializer;
    /**
     * Regularizer function applied to the `embeddings` matrix.
     */
    embeddingsRegularizer?: RegularizerIdentifier | Regularizer;
    /**
     * Regularizer function applied to the activation.
     */
    activityRegularizer?: RegularizerIdentifier | Regularizer;
    /**
     * Constraint function applied to the `embeddings` matrix.
     */
    embeddingsConstraint?: ConstraintIdentifier | Constraint;
    /**
     * Whether the input value 0 is a special "padding" value that should be
     * masked out. This is useful when using recurrent layers which may take
     * variable length input.
     *
     * If this is `True` then all subsequent layers in the model need to support
     * masking or an exception will be raised. If maskZero is set to `True`, as a
     * consequence, index 0 cannot be used in the vocabulary (inputDim should
     * equal size of vocabulary + 1).
     */
    maskZero?: boolean;
    /**
     * Length of input sequences, when it is constant.
     *
     * This argument is required if you are going to connect `flatten` then
     * `dense` layers upstream (without it, the shape of the dense outputs cannot
     * be computed).
     */
    inputLength?: number | number[];
}
declare class Embedding extends Layer {
    /** @nocollapse */
    static className: string;
    private inputDim;
    private outputDim;
    private embeddingsInitializer;
    private maskZero;
    private inputLength;
    private embeddings;
    readonly DEFAULT_EMBEDDINGS_INITIALIZER: InitializerIdentifier;
    private readonly embeddingsRegularizer?;
    private readonly embeddingsConstraint?;
    constructor(args: EmbeddingLayerArgs);
    build(inputShape: Shape | Shape[]): void;
    protected warnOnIncompatibleInputShape(inputShape: Shape): void;
    computeMask(inputs: Tensor | Tensor[], mask?: Tensor | Tensor[]): Tensor;
    computeOutputShape(inputShape: Shape | Shape[]): Shape | Shape[];
    call(inputs: Tensor | Tensor[], kwargs: Kwargs): Tensor | Tensor[];
    getConfig(): serialization.ConfigDict;
}

/// <amd-module name="@tensorflow/tfjs-layers/dist/keras_format/layers/embeddings_serialization" />
interface EmbeddingLayerConfig extends LayerConfig {
    input_dim: number;
    output_dim: number;
    embeddings_initializer?: InitializerSerialization;
    embeddings_regularizer?: RegularizerSerialization;
    activity_regularizer?: RegularizerSerialization;
    embeddings_constraint?: ConstraintSerialization;
    mask_zero?: boolean;
    input_length?: number | number[];
}
declare type EmbeddingLayerSerialization = BaseLayerSerialization<'Embedding', EmbeddingLayerConfig>;
declare type EmbeddingLayerClassName = EmbeddingLayerSerialization['class_name'];
/**
 * A string array of valid EmbeddingLayer class names.
 *
 * This is guaranteed to match the `EmbeddingLayerClassName` union type.
 */
declare const embeddingLayerClassNames: EmbeddingLayerClassName[];

/// <amd-module name="@tensorflow/tfjs-core/dist/engine" />
/**
 * A function that computes an output. The save function is for saving tensors
 * computed in the forward pass, that we need in the backward pass.
 */
declare type ForwardFunc<T> = (backend: KernelBackend, save?: GradSaveFunc) => T;
/**
 * @docalias (a: Tensor, b: Tensor,..., save?: Function) => {
 *   value: Tensor,
 *   gradFunc: (dy: Tensor, saved?: NamedTensorMap) => Tensor | Tensor[]
 * }
 */
declare type CustomGradientFunc<T extends Tensor> = (...inputs: Array<Tensor | GradSaveFunc>) => {
    value: T;
    gradFunc: (dy: T, saved: Tensor[]) => Tensor | Tensor[];
};
declare type MemoryInfo = {
    numTensors: number;
    numDataBuffers: number;
    numBytes: number;
    unreliable?: boolean;
    reasons: string[];
};
declare type KernelInfo = {
    name: string;
    bytesAdded: number;
    totalBytesSnapshot: number;
    tensorsAdded: number;
    totalTensorsSnapshot: number;
    inputShapes: number[][];
    outputShapes: number[][];
    kernelTimeMs: number | {
        error: string;
    } | Promise<number | {
        error: string;
    }>;
    extraInfo: string | Promise<string>;
};
declare type ProfileInfo = {
    newBytes: number;
    newTensors: number;
    peakBytes: number;
    kernels: KernelInfo[];
    result: TensorContainer;
    kernelNames: string[];
};
interface TimingInfo extends BackendTimingInfo {
    wallMs: number;
}
/** @docalias Function */
declare type ScopeFn<T extends TensorContainer> = () => T;
interface ScopeState {
    track: Tensor[];
    name: string;
    id: number;
}
declare class EngineState {
    registeredVariables: NamedVariableMap;
    nextTapeNodeId: number;
    numBytes: number;
    numTensors: number;
    numStringTensors: number;
    numDataBuffers: number;
    activeTape: TapeNode[];
    gradientDepth: number;
    kernelDepth: number;
    activeScope: ScopeState;
    scopeStack: ScopeState[];
    /**
     * Keeps track of the number of data moves during a kernel execution. We
     * maintain a stack since kernels can call other kernels, recursively.
     */
    numDataMovesStack: number[];
    nextScopeId: number;
    tensorInfo: WeakMap<object, {
        backend: KernelBackend;
        bytes: number;
        dtype: DataType;
        shape: number[];
    }>;
    profiling: boolean;
    activeProfile: ProfileInfo;
    dispose(): void;
}
declare class Engine implements TensorTracker, DataMover {
    ENV: Environment;
    state: EngineState;
    backendName: string;
    registry: {
        [id: string]: KernelBackend;
    };
    registryFactory: {
        [id: string]: {
            factory: () => KernelBackend | Promise<KernelBackend>;
            priority: number;
        };
    };
    private profiler;
    private backendInstance;
    private pendingBackendInit;
    private pendingBackendInitId;
    constructor(ENV: Environment);
    ready(): Promise<void>;
    get backend(): KernelBackend;
    backendNames(): string[];
    findBackend(backendName: string): KernelBackend;
    findBackendFactory(backendName: string): () => KernelBackend | Promise<KernelBackend>;
    registerBackend(backendName: string, factory: () => KernelBackend | Promise<KernelBackend>, priority?: number): boolean;
    setBackend(backendName: string): Promise<boolean>;
    private setupRegisteredKernels;
    private disposeRegisteredKernels;
    /**
     * Initializes a backend by looking up the backend name in the factory
     * registry and calling the factory method. Returns a boolean representing
     * whether the initialization of the backend suceeded. Throws an error if
     * there is no backend in the factory registry.
     */
    private initializeBackend;
    removeBackend(backendName: string): void;
    private getSortedBackends;
    private initializeBackendsAndReturnBest;
    moveData(backend: KernelBackend, dataId: DataId): void;
    tidy<T extends TensorContainer>(nameOrFn: string | ScopeFn<T>, fn?: ScopeFn<T>): T;
    private scopedRun;
    private static nextTensorId;
    private nextTensorId;
    private static nextVariableId;
    private nextVariableId;
    /**
     * This method is called instead of the public-facing tensor.clone() when
     * saving a tensor for backwards pass. It makes sure to add the clone
     * operation to the tape regardless of being called inside a kernel
     * execution.
     */
    private clone;
    /**
     * Execute a kernel with the given name and return the output tensor.
     *
     * @param kernelName The name of the kernel to execute.
     * @param inputs A map of input names to tensors.
     * @param attrs A map of attribute names to their values. An attribute is a
     *     primitive (non-tensor) input to the kernel.
     * @param inputsToSave A list of tensors, inputs to save for the backprop
     *     computation.
     * @param outputsToSave A list of booleans, specifying which output to save
     *     for the backprop computation. These are booleans since the output
     * tensors are not visible to the user.
     */
    runKernel<T extends Tensor | Tensor[]>(kernelName: string, inputs: NamedTensorMap, attrs?: NamedAttrMap): T;
    private shouldCheckForMemLeaks;
    private checkKernelForMemLeak;
    /**
     * Internal helper method to execute a kernel Func
     *
     * Use `runKernel` to execute kernels from outside of engine.
     */
    private runKernelFunc;
    /**
     * Saves tensors used in forward mode for use in backward mode.
     *
     * @param tensors the list of tensors to save.
     */
    private saveTensorsForBackwardMode;
    /**
     * Returns a list of tensors to save for a given gradient calculation.
     *
     * @param kernelName name of kernel to look up gradient for.
     * @param inputs a map of input tensors.
     * @param outputs an array of output tensors from forward mode of kernel.
     */
    private getTensorsForGradient;
    /**
     * Internal method used by public APIs for tensor creation. Makes a new
     * tensor with the provided shape, dtype and values. It always
     * creates a new data id and writes the values to the underlying backend.
     */
    makeTensor(values: DataValues, shape: number[], dtype: DataType, backend?: KernelBackend): Tensor;
    /**
     * Internal method used by backends. Makes a new tensor
     * that is a wrapper around an existing data id. It doesn't create
     * a new data id, only increments the ref count used in memory tracking.
     * @deprecated
     */
    makeTensorFromDataId(dataId: DataId, shape: number[], dtype: DataType, backend?: KernelBackend): Tensor;
    /**
     * Internal method used by backends. Makes a new tensor that is a wrapper
     * around an existing data id in TensorInfo. It doesn't create a new data id,
     * only increments the ref count used in memory tracking.
     */
    makeTensorFromTensorInfo(tensorInfo: TensorInfo, backend?: KernelBackend): Tensor;
    makeVariable(initialValue: Tensor, trainable?: boolean, name?: string, dtype?: DataType): Variable;
    trackTensor(a: Tensor, backend: KernelBackend): void;
    incRef(a: Tensor, backend: KernelBackend): void;
    removeDataId(dataId: DataId, backend: KernelBackend): void;
    disposeTensor(a: Tensor): void;
    disposeVariables(): void;
    disposeVariable(v: Variable): void;
    memory(): MemoryInfo;
    profile(query: () => (TensorContainer | Promise<TensorContainer>)): Promise<ProfileInfo>;
    isTapeOn(): boolean;
    private addTapeNode;
    keep<T extends Tensor>(result: T): T;
    private startTape;
    private endTape;
    /**
     * Start a scope. Use this with endScope() to achieve the same functionality
     * as scope() without the need for a function closure.
     */
    startScope(name?: string): void;
    /**
     * End a scope. Use this with startScope() to achieve the same functionality
     * as scope() without the need for a function closure.
     */
    endScope(result?: TensorContainer): void;
    /**
     * Returns gradients of `f` with respect to each of the `xs`. The gradients
     * returned are of the same length as `xs`, but some might be null if `f`
     * was not a function of that `x`. It also takes optional dy to multiply the
     * gradient, which defaults to `1`.
     */
    gradients<T extends Tensor>(f: () => T, xs: Tensor[], dy?: T, allowNoGradients?: boolean): {
        value: T;
        grads: Tensor[];
    };
    customGrad<T extends Tensor>(f: CustomGradientFunc<T>): (...args: Array<Tensor | GradSaveFunc>) => T;
    readSync(dataId: DataId): BackendValues;
    read(dataId: DataId): Promise<BackendValues>;
    readToGPU(dataId: DataId, options?: DataToGPUOptions): GPUData;
    time(query: () => void): Promise<TimingInfo>;
    /**
     * Tracks a Tensor in the current scope to be automatically cleaned up
     * when the current scope ends, and returns the value.
     *
     * @param result The Tensor to track in the current scope.
     */
    private track;
    get registeredVariables(): NamedVariableMap;
    /**
     * Resets the engine state. Removes all backends but does not remove
     * registered backend factories.
     */
    reset(): void;
}
declare function getOrMakeEngine(): Engine;
declare const ENGINE: Engine;
/**
 * A implementation of the add op for use within engine and tape.
 *
 * This allows us to avoid a circular dependency between add.ts and engine.
 * It is exported to be available in tape tests.
 */
declare function add(a: Tensor, b: Tensor): Tensor;
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/engine_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/environment" />
declare type FlagValue = number | boolean;
declare type FlagEvaluationFn = (() => FlagValue) | (() => Promise<FlagValue>);
declare type Flags = {
    [featureName: string]: FlagValue;
};
declare type FlagRegistryEntry = {
    evaluationFn: FlagEvaluationFn;
    setHook?: (value: FlagValue) => void;
};
/**
 * The environment contains evaluated flags as well as the registered platform.
 * This is always used as a global singleton and can be retrieved with
 * `tf.env()`.
 *
 * @doc {heading: 'Environment'}
 */
declare class Environment {
    global: any;
    private flags;
    private flagRegistry;
    private urlFlags;
    platformName: string;
    platform: Platform;
    getQueryParams: typeof getQueryParams;
    constructor(global: any);
    setPlatform(platformName: string, platform: Platform): void;
    registerFlag(flagName: string, evaluationFn: FlagEvaluationFn, setHook?: (value: FlagValue) => void): void;
    getAsync(flagName: string): Promise<FlagValue>;
    get(flagName: string): FlagValue;
    getNumber(flagName: string): number;
    getBool(flagName: string): boolean;
    getFlags(): Flags;
    get features(): Flags;
    set(flagName: string, value: FlagValue): void;
    private evaluateFlag;
    setFlags(flags: Flags): void;
    reset(): void;
    private populateURLFlags;
}
declare function getQueryParams(queryString: string): {
    [key: string]: string;
};
/**
 * Returns the current environment (a global singleton).
 *
 * The environment object contains the evaluated feature values as well as the
 * active platform.
 *
 * @doc {heading: 'Environment'}
 */
declare function env(): Environment;
declare let ENV: Environment;
declare function setEnvironmentGlobal(environment: Environment): void;
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/environment_test" />
{ };
/// <amd-module name="@tensorflow/tfjs-core/dist/ops/equal" />
/**
 * Returns the truth value of (a == b) element-wise. Supports broadcasting.
 *
 * ```js
 * const a = tf.tensor1d([1, 2, 3]);
 * const b = tf.tensor1d([2, 2, 2]);
 *
 * a.equal(b).print();
 * ```
 *
 * @param a The first input tensor.
 * @param b The second input tensor. Must have the same dtype as `a`.
 *
 * @doc {heading: 'Operations', subheading: 'Logical'}
 */
declare function equal_<T extends Tensor>(a: Tensor | TensorLike, b: Tensor | TensorLike): T;
declare const equal: typeof equal_;
{ };
/// <amd-module name="@tensorflow/tfjs-core/dist/ops/equal_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/erf" />
/**
 * Computes Gauss error function of the input `tf.Tensor` element-wise:
 * `erf(x)`
 *
 * ```js
 * const x = tf.tensor1d([0, .1, -.1, .7]);
 *
 * x.erf().print(); // or tf.erf(x);
 * ```
 * @param x The input tensor.
 *
 * @doc {heading: 'Operations', subheading: 'Basic math'}
 */
declare function erf_<T extends Tensor>(x: T | TensorLike): T;
declare const erf: typeof erf_;
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/gradients/Erf_grad" />
declare const erfGradConfig: GradConfig;

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/erf_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/erf_util" />
declare const ERF_P = 0.3275911;
declare const ERF_A1 = 0.254829592;
declare const ERF_A2 = -0.284496736;
declare const ERF_A3 = 1.421413741;
declare const ERF_A4 = -1.453152027;
declare const ERF_A5 = 1.061405429;

/// <amd-module name="@tensorflow/tfjs-layers/dist/errors" />
/**
 * Explicit error types.
 *
 * See the following link for more information about why the code includes
 * calls to setPrototypeOf:
 *
 * https://github.com/Microsoft/TypeScript-wiki/blob/master/Breaking-Changes.md#extending-built-ins-like-error-array-and-map-may-no-longer-work
 */
/**
 * Equivalent of Python's AttributeError.
 */
declare class AttributeError extends Error {
    constructor(message?: string);
}
/**
 * Equivalent of Python's RuntimeError.
 */
declare class RuntimeError extends Error {
    constructor(message?: string);
}
/**
 * Equivalent of Python's ValueError.
 */
declare class ValueError extends Error {
    constructor(message?: string);
}
/**
 * Equivalent of Python's NotImplementedError.
 */
declare class NotImplementedError extends Error {
    constructor(message?: string);
}
/**
 * Equivalent of Python's AssertionError.
 */
declare class AssertionError extends Error {
    constructor(message?: string);
}
/**
 * Equivalent of Python's IndexError.
 */
declare class IndexError extends Error {
    constructor(message?: string);
}

/// <amd-module name="@tensorflow/tfjs-core/dist/public/chained_ops/euclidean_norm" />
declare module '../../tensor' {
    interface Tensor<R extends Rank = Rank> {
        euclideanNorm<T extends Tensor>(this: T, axis?: number | number[], keepDims?: boolean): T;
    }
}

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/euclidean_norm_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-layers/dist/engine/executor" />
/**
 * Executor: Evaluates SymbolicTensor based on feeds.
 */
/**
 * A concrete Tensor value for a symbolic tensor as the key.
 */
interface Feed {
    key: SymbolicTensor;
    value: Tensor;
}
/**
 * FeedDict: A mapping from unique SymbolicTensors to feed values for them.
 * A feed value is a concrete value represented as an `Tensor`.
 */
declare class FeedDict {
    private id2Value;
    private id2Mask;
    private name2Id;
    /**
     * Constructor, optionally does copy-construction.
     * @param feeds An Array of `Feed`s, or another `FeedDict`, in which case
     *   copy-construction will be performed.
     */
    constructor(feeds?: Feed[] | FeedDict);
    /**
     * Add a key-value pair to the FeedDict.
     *
     * @param key The key of the feed.
     * @param value The value of the tensor feed.
     * @param mask The value of the mask feed (optional).
     * @returns This `FeedDict`.
     * @throws ValueError: If the key `SymbolicTensor` already exists in the
     *   `FeedDict`.
     */
    add(key: SymbolicTensor, value: Tensor, mask?: Tensor): FeedDict;
    /**
     * Add a Feed to the FeedDict.
     * @param feed The new `Feed` to add.
     * @returns This `FeedDict`.
     */
    addFeed(feed: Feed): void;
    /**
     * Probe whether a key already exists in the FeedDict.
     * @param key
     */
    hasKey(key: SymbolicTensor): boolean;
    /**
     * Get all the SymbolicTensor available in this FeedDict.
     */
    names(): string[];
    /**
     * Get the feed value for given key.
     * @param key The SymbolicTensor, or its name (as a string), of which the
     *     value is sought.
     * @returns If `key` exists, the corresponding feed value.
     * @throws ValueError: If `key` does not exist in this `FeedDict`.
     */
    getValue(key: SymbolicTensor | string): Tensor;
    /**
     * Get the feed mask for given key.
     * @param key The SymbolicTensor, or its name (as a string), of which the
     *     value is sought.
     * @returns If `key` exists, the corresponding feed mask.
     * @throws ValueError: If `key` does not exist in this `FeedDict`.
     */
    getMask(key: SymbolicTensor | string): Tensor;
    /** Dispose all mask Tensors held by this object. */
    disposeMasks(): void;
}
declare const cachedSorted: LruCache<SymbolicTensor[]>;
declare const cachedRecipientCounts: LruCache<RecipientCounts>;
declare function updateCacheMaxEntries(maxEntries: number): void;
/**
 * Interface for the optional object used for probing the memory
 * usage and other statistics during execution.
 */
interface ExecutionProbe {
    /**
     * Maximum number of tensors that exist during all steps of the
     * execution. Tensor counts are measured at the beginning of every
     * step.
     */
    maxNumTensors?: number;
    /**
     * Minimum number of tensors that exist during all steps of the
     * execution. Tensor counts are measured at the beginning of every
     * step.
     */
    minNumTensors?: number;
}
/**
 * Execute a SymbolicTensor by using concrete feed values.
 *
 * A `SymbolicTensor` object is a node in a computation graph of TF.js
 * Layers. The object is backed by a source layer and input
 * `SymbolicTensor`s to the source layer. This method evaluates
 * the `call()` method of the source layer, using concrete values of the
 * inputs obtained from either
 * * `feedDict`, if the input key exists in `feedDict`, or else,
 * * a recursive call to `execute()` itself.
 *
 * @param x: The `SymbolicTensor` to execute.
 * @param feedDict: The feed values, as base condition of the recursion.
 *   execution.
 * @param kwargs: Optional keyword arguments.
 * @param probe: A probe object (of interface `ExecutionProbe`) used for
 *   testing memory footprint of `execute` calls.
 * @returns Result of the execution.
 * @throws ValueError: If any `SymbolicTensor`s from `InputLayer`s
 *   encountered during the execution lacks a feed value in `feedDict`.
 */
declare function execute(fetches: SymbolicTensor | SymbolicTensor[], feedDict: FeedDict, kwargs?: Kwargs, probe?: ExecutionProbe): Tensor | Tensor[] | [Tensor | Tensor[]];
declare type RecipientCounts = {
    [fetchName: string]: number;
};
declare type RecipientMap = {
    [fetchName: string]: Set<string>;
};
/**
 * Sort the `SymbolicTensor`s topologically, for a single fetch.
 *
 * This helper function processes the upstream SymbolicTensors of a single
 * fetch.
 *
 * @param fetch The single fetch requested.
 * @param feedDict The dictionary of fed values.
 * @returns sorted: Topologically-sorted array of SymbolicTensors.
 *   recipientMap: Recipient names for all SymbolicTensors in `sorted`.
 */
declare function getTopologicalSortAndRecipientCountsForOneFetch(fetch: SymbolicTensor, feedDict: FeedDict): {
    sorted: SymbolicTensor[];
    recipientMap: RecipientMap;
};
{ };

/**
 * LruCache: A mapping from the String to T. If the number of the entries is
 * exceeding the `maxEntries`, the LruCache will delete the least recently
 * used entry.
 */
/// <amd-module name="@tensorflow/tfjs-layers/dist/utils/executor_utils" />
declare class LruCache<T> {
    private cache;
    private maxEntries;
    constructor(maxEntries?: number);
    /**
     * Get the entry for the key and mark it as used recently.
     */
    get(key: string): T;
    /**
     * Put the entry into the cache. If the key already existed, mark the key as
     * used recently.
     */
    put(key: string, value: T): void;
    /**
     * Get the MaxEntries of the cache.
     */
    getMaxEntries(): number;
    /**
     * Set the MaxEntries of the cache. If the maxEntries is decreased, reduce
     * entries in the cache.
     */
    setMaxEntries(maxEntries: number): void;
}

/// <amd-module name="@tensorflow/tfjs-core/dist/public/chained_ops/exp" />
declare module '../../tensor' {
    interface Tensor<R extends Rank = Rank> {
        exp<T extends Tensor>(this: T): T;
    }
}

/// <amd-module name="@tensorflow/tfjs-core/dist/gradients/ExpandDims_grad" />
declare const expandDimsGradConfig: GradConfig;

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/expand_dims" />
/**
 * Returns a `tf.Tensor` that has expanded rank, by inserting a dimension
 * into the tensor's shape.
 *
 * ```js
 * const x = tf.tensor1d([1, 2, 3, 4]);
 * const axis = 1;
 * x.expandDims(axis).print();
 * ```
 *
 * @param x The input tensor whose dimensions are to be expanded.
 * @param axis The dimension index at which to insert shape of `1`. Defaults
 *     to 0 (the first dimension).
 *
 * @doc {heading: 'Tensors', subheading: 'Transformations'}
 */
declare function expandDims_<T extends Tensor>(x: Tensor | TensorLike, axis?: number): T;
declare const expandDims: typeof expandDims_;
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/expand_dims_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/expm1" />
/**
 * Computes exponential of the input `tf.Tensor` minus one element-wise.
 * `e ^ x - 1`
 *
 * ```js
 * const x = tf.tensor1d([1, 2, -3]);
 *
 * x.expm1().print();  // or tf.expm1(x)
 * ```
 * @param x The input tensor.
 *
 * @doc {heading: 'Operations', subheading: 'Basic math'}
 */
declare function expm1_<T extends Tensor>(x: T | TensorLike): T;
declare const expm1: typeof expm1_;
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/gradients/Expm1_grad" />
declare const expm1GradConfig: GradConfig;

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/expm1_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-layers/dist/exports" />
/**
 * Exported functions.
 */
/**
 * A model is a data structure that consists of `Layers` and defines inputs
 * and outputs.
 *
 * The key difference between `tf.model` and `tf.sequential` is that
 * `tf.model` is more generic, supporting an arbitrary graph (without
 * cycles) of layers. `tf.sequential` is less generic and supports only a linear
 * stack of layers.
 *
 * When creating a `tf.LayersModel`, specify its input(s) and output(s). Layers
 * are used to wire input(s) to output(s).
 *
 * For example, the following code snippet defines a model consisting of
 * two `dense` layers, with 10 and 4 units, respectively.
 *
 * ```js
 * // Define input, which has a size of 5 (not including batch dimension).
 * const input = tf.input({shape: [5]});
 *
 * // First dense layer uses relu activation.
 * const denseLayer1 = tf.layers.dense({units: 10, activation: 'relu'});
 * // Second dense layer uses softmax activation.
 * const denseLayer2 = tf.layers.dense({units: 4, activation: 'softmax'});
 *
 * // Obtain the output symbolic tensor by applying the layers on the input.
 * const output = denseLayer2.apply(denseLayer1.apply(input));
 *
 * // Create the model based on the inputs.
 * const model = tf.model({inputs: input, outputs: output});
 *
 * // The model can be used for training, evaluation and prediction.
 * // For example, the following line runs prediction with the model on
 * // some fake data.
 * model.predict(tf.ones([2, 5])).print();
 * ```
 * See also:
 *   `tf.sequential`, `tf.loadLayersModel`.
 *
 * @doc {heading: 'Models', subheading: 'Creation'}
 */
declare function model(args: ContainerArgs): LayersModel;
/**
 * Creates a `tf.Sequential` model.  A sequential model is any model where the
 * outputs of one layer are the inputs to the next layer, i.e. the model
 * topology is a simple 'stack' of layers, with no branching or skipping.
 *
 * This means that the first layer passed to a `tf.Sequential` model should have
 * a defined input shape. What that means is that it should have received an
 * `inputShape` or `batchInputShape` argument, or for some type of layers
 * (recurrent, Dense...) an `inputDim` argument.
 *
 * The key difference between `tf.model` and `tf.sequential` is that
 * `tf.sequential` is less generic, supporting only a linear stack of layers.
 * `tf.model` is more generic and supports an arbitrary graph (without
 * cycles) of layers.
 *
 * Examples:
 *
 * ```js
 * const model = tf.sequential();
 *
 * // First layer must have an input shape defined.
 * model.add(tf.layers.dense({units: 32, inputShape: [50]}));
 * // Afterwards, TF.js does automatic shape inference.
 * model.add(tf.layers.dense({units: 4}));
 *
 * // Inspect the inferred shape of the model's output, which equals
 * // `[null, 4]`. The 1st dimension is the undetermined batch dimension; the
 * // 2nd is the output size of the model's last layer.
 * console.log(JSON.stringify(model.outputs[0].shape));
 * ```
 *
 * It is also possible to specify a batch size (with potentially undetermined
 * batch dimension, denoted by "null") for the first layer using the
 * `batchInputShape` key. The following example is equivalent to the above:
 *
 * ```js
 * const model = tf.sequential();
 *
 * // First layer must have a defined input shape
 * model.add(tf.layers.dense({units: 32, batchInputShape: [null, 50]}));
 * // Afterwards, TF.js does automatic shape inference.
 * model.add(tf.layers.dense({units: 4}));
 *
 * // Inspect the inferred shape of the model's output.
 * console.log(JSON.stringify(model.outputs[0].shape));
 * ```
 *
 * You can also use an `Array` of already-constructed `Layer`s to create
 * a `tf.Sequential` model:
 *
 * ```js
 * const model = tf.sequential({
 *   layers: [tf.layers.dense({units: 32, inputShape: [50]}),
 *            tf.layers.dense({units: 4})]
 * });
 * console.log(JSON.stringify(model.outputs[0].shape));
 * ```
 *
 * @doc {heading: 'Models', subheading: 'Creation'}
 */
declare function sequential(config?: SequentialArgs): Sequential;
/**
 * Used to instantiate an input to a model as a `tf.SymbolicTensor`.
 *
 * Users should call the `input` factory function for
 * consistency with other generator functions.
 *
 * Example:
 *
 * ```js
 * // Defines a simple logistic regression model with 32 dimensional input
 * // and 3 dimensional output.
 * const x = tf.input({shape: [32]});
 * const y = tf.layers.dense({units: 3, activation: 'softmax'}).apply(x);
 * const model = tf.model({inputs: x, outputs: y});
 * model.predict(tf.ones([2, 32])).print();
 * ```
 *
 * Note: `input` is only necessary when using `model`. When using
 * `sequential`, specify `inputShape` for the first layer or use `inputLayer`
 * as the first layer.
 *
 * @doc {heading: 'Models', subheading: 'Inputs'}
 */
declare function input(config: InputConfig): SymbolicTensor;
declare function registerCallbackConstructor(verbosityLevel: number, callbackConstructor: BaseCallbackConstructor): void;
/// <amd-module name="@tensorflow/tfjs-layers/dist/exports_constraints" />

/**
 * MaxNorm weight constraint.
 *
 * Constrains the weights incident to each hidden unit
 * to have a norm less than or equal to a desired value.
 *
 * References
 *       - [Dropout: A Simple Way to Prevent Neural Networks from Overfitting
 * Srivastava, Hinton, et al.
 * 2014](http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)
 *
 * @doc {heading: 'Constraints',namespace: 'constraints'}
 */
declare function maxNorm(args: MaxNormArgs): Constraint;
/**
 * Constrains the weights incident to each hidden unit to have unit norm.
 *
 * @doc {heading: 'Constraints', namespace: 'constraints'}
 */
declare function unitNorm(args: UnitNormArgs): Constraint;
/**
 * Constrains the weight to be non-negative.
 *
 * @doc {heading: 'Constraints', namespace: 'constraints'}
 */
declare function nonNeg(): Constraint;
/** @doc {heading: 'Constraints', namespace: 'constraints'} */
declare function minMaxNorm(config: MinMaxNormArgs): Constraint;
/// <amd-module name="@tensorflow/tfjs-layers/dist/exports_initializers" />

/**
 * Initializer that generates tensors initialized to 0.
 *
 * @doc {heading: 'Initializers', namespace: 'initializers'}
 */
declare function zeros(): Zeros;
/**
 * Initializer that generates tensors initialized to 1.
 *
 * @doc {heading: 'Initializers', namespace: 'initializers'}
 */
declare function ones(): Initializer;
/**
 * Initializer that generates values initialized to some constant.
 *
 * @doc {heading: 'Initializers', namespace: 'initializers'}
 */
declare function constant(args: ConstantArgs): Initializer;
/**
 * Initializer that generates random values initialized to a uniform
 * distribution.
 *
 * Values will be distributed uniformly between the configured minval and
 * maxval.
 *
 * @doc {heading: 'Initializers', namespace: 'initializers'}
 */
declare function randomUniform(args: RandomUniformArgs): Initializer;
/**
 * Initializer that generates random values initialized to a normal
 * distribution.
 *
 * @doc {heading: 'Initializers', namespace: 'initializers'}
 */
declare function randomNormal(args: RandomNormalArgs): Initializer;
/**
 * Initializer that generates random values initialized to a truncated normal
 * distribution.
 *
 * These values are similar to values from a `RandomNormal` except that values
 * more than two standard deviations from the mean are discarded and re-drawn.
 * This is the recommended initializer for neural network weights and filters.
 *
 * @doc {heading: 'Initializers', namespace: 'initializers'}
 */
declare function truncatedNormal(args: TruncatedNormalArgs): Initializer;
/**
 * Initializer that generates the identity matrix.
 * Only use for square 2D matrices.
 *
 * @doc {heading: 'Initializers', namespace: 'initializers'}
 */
declare function identity(args: IdentityArgs): Initializer;
/**
 * Initializer capable of adapting its scale to the shape of weights.
 * With distribution=NORMAL, samples are drawn from a truncated normal
 * distribution centered on zero, with `stddev = sqrt(scale / n)` where n is:
 *   - number of input units in the weight tensor, if mode = FAN_IN.
 *   - number of output units, if mode = FAN_OUT.
 *   - average of the numbers of input and output units, if mode = FAN_AVG.
 * With distribution=UNIFORM,
 * samples are drawn from a uniform distribution
 * within [-limit, limit], with `limit = sqrt(3 * scale / n)`.
 *
 * @doc {heading: 'Initializers',namespace: 'initializers'}
 */
declare function varianceScaling(config: VarianceScalingArgs): Initializer;
/**
 * Glorot uniform initializer, also called Xavier uniform initializer.
 * It draws samples from a uniform distribution within [-limit, limit]
 * where `limit` is `sqrt(6 / (fan_in + fan_out))`
 * where `fan_in` is the number of input units in the weight tensor
 * and `fan_out` is the number of output units in the weight tensor
 *
 * Reference:
 *   Glorot & Bengio, AISTATS 2010
 *       http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf.
 *
 * @doc {heading: 'Initializers', namespace: 'initializers'}
 */
declare function glorotUniform(args: SeedOnlyInitializerArgs): Initializer;
/**
 * Glorot normal initializer, also called Xavier normal initializer.
 * It draws samples from a truncated normal distribution centered on 0
 * with `stddev = sqrt(2 / (fan_in + fan_out))`
 * where `fan_in` is the number of input units in the weight tensor
 * and `fan_out` is the number of output units in the weight tensor.
 *
 * Reference:
 *   Glorot & Bengio, AISTATS 2010
 *       http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf
 *
 * @doc {heading: 'Initializers', namespace: 'initializers'}
 */
declare function glorotNormal(args: SeedOnlyInitializerArgs): Initializer;
/**
 * He normal initializer.
 *
 * It draws samples from a truncated normal distribution centered on 0
 * with `stddev = sqrt(2 / fanIn)`
 * where `fanIn` is the number of input units in the weight tensor.
 *
 * Reference:
 *     He et al., http://arxiv.org/abs/1502.01852
 *
 * @doc {heading: 'Initializers', namespace: 'initializers'}
 */
declare function heNormal(args: SeedOnlyInitializerArgs): Initializer;
/**
 * He uniform initializer.
 *
 * It draws samples from a uniform distribution within [-limit, limit]
 * where `limit` is `sqrt(6 / fan_in)`
 * where `fanIn` is the number of input units in the weight tensor.
 *
 * Reference:
 *     He et al., http://arxiv.org/abs/1502.01852
 *
 * @doc {heading: 'Initializers',namespace: 'initializers'}
 */
declare function heUniform(args: SeedOnlyInitializerArgs): Initializer;
/**
 * LeCun normal initializer.
 *
 * It draws samples from a truncated normal distribution centered on 0
 * with `stddev = sqrt(1 / fanIn)`
 * where `fanIn` is the number of input units in the weight tensor.
 *
 * References:
 *   [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)
 *   [Efficient Backprop](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf)
 *
 * @doc {heading: 'Initializers', namespace: 'initializers'}
 */
declare function leCunNormal(args: SeedOnlyInitializerArgs): Initializer;
/**
 * LeCun uniform initializer.
 *
 * It draws samples from a uniform distribution in the interval
 * `[-limit, limit]` with `limit = sqrt(3 / fanIn)`,
 * where `fanIn` is the number of input units in the weight tensor.
 *
 * @doc {heading: 'Initializers', namespace: 'initializers'}
 */
declare function leCunUniform(args: SeedOnlyInitializerArgs): Initializer;
/**
 * Initializer that generates a random orthogonal matrix.
 *
 * Reference:
 * [Saxe et al., http://arxiv.org/abs/1312.6120](http://arxiv.org/abs/1312.6120)
 *
 * @doc {heading: 'Initializers', namespace: 'initializers'}
 */
declare function orthogonal(args: OrthogonalArgs): Initializer;

/// <amd-module name="@tensorflow/tfjs-layers/dist/exports_layers" />
/**
 * An input layer is an entry point into a `tf.LayersModel`.
 *
 * `InputLayer` is generated automatically for `tf.Sequential` models by
 * specifying the `inputshape` or `batchInputShape` for the first layer.  It
 * should not be specified explicitly. However, it can be useful sometimes,
 * e.g., when constructing a sequential model from a subset of another
 * sequential model's layers. Like the code snippet below shows.
 *
 * ```js
 * // Define a model which simply adds two inputs.
 * const model1 = tf.sequential();
 * model1.add(tf.layers.dense({inputShape: [4], units: 3, activation: 'relu'}));
 * model1.add(tf.layers.dense({units: 1, activation: 'sigmoid'}));
 * model1.summary();
 * model1.predict(tf.zeros([1, 4])).print();
 *
 * // Construct another model, reusing the second layer of `model1` while
 * // not using the first layer of `model1`. Note that you cannot add the second
 * // layer of `model` directly as the first layer of the new sequential model,
 * // because doing so will lead to an error related to the fact that the layer
 * // is not an input layer. Instead, you need to create an `inputLayer` and add
 * // it to the new sequential model before adding the reused layer.
 * const model2 = tf.sequential();
 * // Use an inputShape that matches the input shape of `model1`'s second
 * // layer.
 * model2.add(tf.layers.inputLayer({inputShape: [3]}));
 * model2.add(model1.layers[1]);
 * model2.summary();
 * model2.predict(tf.zeros([1, 3])).print();
 * ```
 *
 * @doc {heading: 'Layers', subheading: 'Inputs', namespace: 'layers'}
 */
declare function inputLayer(args: InputLayerArgs): InputLayer;
/**
 * Exponential Linear Unit (ELU).
 *
 * It follows:
 * `f(x) =  alpha * (exp(x) - 1.) for x < 0`,
 * `f(x) = x for x >= 0`.
 *
 * Input shape:
 *   Arbitrary. Use the configuration `inputShape` when using this layer as the
 *   first layer in a model.
 *
 * Output shape:
 *   Same shape as the input.
 *
 * References:
 *   - [Fast and Accurate Deep Network Learning by Exponential Linear Units
 * (ELUs)](https://arxiv.org/abs/1511.07289v1)
 *
 * @doc {
 *   heading: 'Layers',
 *   subheading: 'Advanced Activation',
 *   namespace: 'layers'
 * }
 */
declare function elu(args?: ELULayerArgs): ELU;
/**
 * Rectified Linear Unit activation function.
 *
 * Input shape:
 *   Arbitrary. Use the config field `inputShape` (Array of integers, does
 *   not include the sample axis) when using this layer as the first layer
 *   in a model.
 *
 * Output shape:
 *   Same shape as the input.
 *
 * @doc {
 *   heading: 'Layers',
 *   subheading: 'Advanced Activation',
 *   namespace: 'layers'
 * }
 */
declare function reLU(args?: ReLULayerArgs): ReLU;
/**
 * Leaky version of a rectified linear unit.
 *
 * It allows a small gradient when the unit is not active:
 * `f(x) = alpha * x for x < 0.`
 * `f(x) = x for x >= 0.`
 *
 * Input shape:
 *   Arbitrary. Use the configuration `inputShape` when using this layer as the
 *   first layer in a model.
 *
 * Output shape:
 *   Same shape as the input.
 *
 * @doc {
 *   heading: 'Layers',
 *   subheading: 'Advanced Activation',
 *   namespace: 'layers'
 * }
 */
declare function leakyReLU(args?: LeakyReLULayerArgs): LeakyReLU;
/**
 * Parameterized version of a leaky rectified linear unit.
 *
 * It follows
 * `f(x) = alpha * x for x < 0.`
 * `f(x) = x for x >= 0.`
 * wherein `alpha` is a trainable weight.
 *
 * Input shape:
 *   Arbitrary. Use the configuration `inputShape` when using this layer as the
 *   first layer in a model.
 *
 * Output shape:
 *   Same shape as the input.
 *
 * @doc {
 *   heading: 'Layers',
 *   subheading: 'Advanced Activation',
 *   namespace: 'layers'
 * }
 */
declare function prelu(args?: PReLULayerArgs): PReLU;
/**
 * Softmax activation layer.
 *
 * Input shape:
 *   Arbitrary. Use the configuration `inputShape` when using this layer as the
 *   first layer in a model.
 *
 * Output shape:
 *   Same shape as the input.
 *
 * @doc {
 *   heading: 'Layers',
 *   subheading: 'Advanced Activation',
 *   namespace: 'layers'
 * }
 */
declare function softmax(args?: SoftmaxLayerArgs): Softmax;
/**
 * Thresholded Rectified Linear Unit.
 *
 * It follows:
 * `f(x) = x for x > theta`,
 * `f(x) = 0 otherwise`.
 *
 * Input shape:
 *   Arbitrary. Use the configuration `inputShape` when using this layer as the
 *   first layer in a model.
 *
 * Output shape:
 *   Same shape as the input.
 *
 * References:
 *   - [Zero-Bias Autoencoders and the Benefits of Co-Adapting
 * Features](http://arxiv.org/abs/1402.3337)
 *
 * @doc {
 *   heading: 'Layers',
 *   subheading: 'Advanced Activation',
 *   namespace: 'layers'
 * }
 */
declare function thresholdedReLU(args?: ThresholdedReLULayerArgs): ThresholdedReLU;
/**
 * 1D convolution layer (e.g., temporal convolution).
 *
 * This layer creates a convolution kernel that is convolved
 * with the layer input over a single spatial (or temporal) dimension
 * to produce a tensor of outputs.
 *
 * If `use_bias` is True, a bias vector is created and added to the outputs.
 *
 * If `activation` is not `null`, it is applied to the outputs as well.
 *
 * When using this layer as the first layer in a model, provide an
 * `inputShape` argument `Array` or `null`.
 *
 * For example, `inputShape` would be:
 * - `[10, 128]` for sequences of 10 vectors of 128-dimensional vectors
 * - `[null, 128]` for variable-length sequences of 128-dimensional vectors.
 *
 * @doc {heading: 'Layers', subheading: 'Convolutional',  namespace: 'layers'}
 */
declare function conv1d(args: ConvLayerArgs): Conv1D;
/**
 * 2D convolution layer (e.g. spatial convolution over images).
 *
 * This layer creates a convolution kernel that is convolved
 * with the layer input to produce a tensor of outputs.
 *
 * If `useBias` is True, a bias vector is created and added to the outputs.
 *
 * If `activation` is not `null`, it is applied to the outputs as well.
 *
 * When using this layer as the first layer in a model,
 * provide the keyword argument `inputShape`
 * (Array of integers, does not include the sample axis),
 * e.g. `inputShape=[128, 128, 3]` for 128x128 RGB pictures
 * in `dataFormat='channelsLast'`.
 *
 * @doc {heading: 'Layers', subheading: 'Convolutional', namespace: 'layers'}
 */
declare function conv2d(args: ConvLayerArgs): Conv2D;
/**
 * Transposed convolutional layer (sometimes called Deconvolution).
 *
 * The need for transposed convolutions generally arises
 * from the desire to use a transformation going in the opposite direction of
 * a normal convolution, i.e., from something that has the shape of the output
 * of some convolution to something that has the shape of its input while
 * maintaining a connectivity pattern that is compatible with said
 * convolution.
 *
 * When using this layer as the first layer in a model, provide the
 * configuration `inputShape` (`Array` of integers, does not include the
 * sample axis), e.g., `inputShape: [128, 128, 3]` for 128x128 RGB pictures in
 * `dataFormat: 'channelsLast'`.
 *
 * Input shape:
 *   4D tensor with shape:
 *   `[batch, channels, rows, cols]` if `dataFormat` is `'channelsFirst'`.
 *   or 4D tensor with shape
 *   `[batch, rows, cols, channels]` if `dataFormat` is `'channelsLast'`.
 *
 * Output shape:
 *   4D tensor with shape:
 *   `[batch, filters, newRows, newCols]` if `dataFormat` is
 * `'channelsFirst'`. or 4D tensor with shape:
 *   `[batch, newRows, newCols, filters]` if `dataFormat` is `'channelsLast'`.
 *
 * References:
 *   - [A guide to convolution arithmetic for deep
 * learning](https://arxiv.org/abs/1603.07285v1)
 *   - [Deconvolutional
 * Networks](http://www.matthewzeiler.com/pubs/cvpr2010/cvpr2010.pdf)
 *
 * @doc {heading: 'Layers', subheading: 'Convolutional', namespace: 'layers'}
 */
declare function conv2dTranspose(args: ConvLayerArgs): Conv2DTranspose;
/**
 * 3D convolution layer (e.g. spatial convolution over volumes).
 *
 * This layer creates a convolution kernel that is convolved
 * with the layer input to produce a tensor of outputs.
 *
 * If `useBias` is True, a bias vector is created and added to the outputs.
 *
 * If `activation` is not `null`, it is applied to the outputs as well.
 *
 * When using this layer as the first layer in a model,
 * provide the keyword argument `inputShape`
 * (Array of integers, does not include the sample axis),
 * e.g. `inputShape=[128, 128, 128, 1]` for 128x128x128 grayscale volumes
 * in `dataFormat='channelsLast'`.
 *
 * @doc {heading: 'Layers', subheading: 'Convolutional', namespace: 'layers'}
 */
declare function conv3d(args: ConvLayerArgs): Conv3D;
declare function conv3dTranspose(args: ConvLayerArgs): Layer;
/**
 * Depthwise separable 2D convolution.
 *
 * Separable convolution consists of first performing
 * a depthwise spatial convolution
 * (which acts on each input channel separately)
 * followed by a pointwise convolution which mixes together the resulting
 * output channels. The `depthMultiplier` argument controls how many
 * output channels are generated per input channel in the depthwise step.
 *
 * Intuitively, separable convolutions can be understood as
 * a way to factorize a convolution kernel into two smaller kernels,
 * or as an extreme version of an Inception block.
 *
 * Input shape:
 *   4D tensor with shape:
 *     `[batch, channels, rows, cols]` if data_format='channelsFirst'
 *   or 4D tensor with shape:
 *     `[batch, rows, cols, channels]` if data_format='channelsLast'.
 *
 * Output shape:
 *   4D tensor with shape:
 *     `[batch, filters, newRows, newCols]` if data_format='channelsFirst'
 *   or 4D tensor with shape:
 *     `[batch, newRows, newCols, filters]` if data_format='channelsLast'.
 *     `rows` and `cols` values might have changed due to padding.
 *
 * @doc {heading: 'Layers', subheading: 'Convolutional', namespace: 'layers'}
 */
declare function separableConv2d(args: SeparableConvLayerArgs): SeparableConv2D;
/**
 * Cropping layer for 2D input (e.g., image).
 *
 * This layer can crop an input
 * at the top, bottom, left and right side of an image tensor.
 *
 * Input shape:
 *   4D tensor with shape:
 *   - If `dataFormat` is `"channelsLast"`:
 *     `[batch, rows, cols, channels]`
 *   - If `data_format` is `"channels_first"`:
 *     `[batch, channels, rows, cols]`.
 *
 * Output shape:
 *   4D with shape:
 *   - If `dataFormat` is `"channelsLast"`:
 *     `[batch, croppedRows, croppedCols, channels]`
 *    - If `dataFormat` is `"channelsFirst"`:
 *     `[batch, channels, croppedRows, croppedCols]`.
 *
 * Examples
 * ```js
 *
 * const model = tf.sequential();
 * model.add(tf.layers.cropping2D({cropping:[[2, 2], [2, 2]],
 *                                inputShape: [128, 128, 3]}));
 * //now output shape is [batch, 124, 124, 3]
 * ```
 *
 * @doc {heading: 'Layers', subheading: 'Convolutional', namespace: 'layers'}
 */
declare function cropping2D(args: Cropping2DLayerArgs): Cropping2D;
/**
 * Upsampling layer for 2D inputs.
 *
 * Repeats the rows and columns of the data
 * by size[0] and size[1] respectively.
 *
 *
 * Input shape:
 *    4D tensor with shape:
 *     - If `dataFormat` is `"channelsLast"`:
 *         `[batch, rows, cols, channels]`
 *     - If `dataFormat` is `"channelsFirst"`:
 *        `[batch, channels, rows, cols]`
 *
 * Output shape:
 *     4D tensor with shape:
 *     - If `dataFormat` is `"channelsLast"`:
 *        `[batch, upsampledRows, upsampledCols, channels]`
 *     - If `dataFormat` is `"channelsFirst"`:
 *         `[batch, channels, upsampledRows, upsampledCols]`
 *
 *
 * @doc {heading: 'Layers', subheading: 'Convolutional', namespace: 'layers'}
 */
declare function upSampling2d(args: UpSampling2DLayerArgs): UpSampling2D;
/**
 * Depthwise separable 2D convolution.
 *
 * Depthwise Separable convolutions consists in performing just the first step
 * in a depthwise spatial convolution (which acts on each input channel
 * separately). The `depthMultiplier` argument controls how many output channels
 * are generated per input channel in the depthwise step.
 *
 * @doc {heading: 'Layers', subheading: 'Convolutional', namespace: 'layers'}
 */
declare function depthwiseConv2d(args: DepthwiseConv2DLayerArgs): DepthwiseConv2D;
/**
 * Applies an activation function to an output.
 *
 * This layer applies element-wise activation function.  Other layers, notably
 * `dense` can also apply activation functions.  Use this isolated activation
 * function to extract the values before and after the
 * activation. For instance:
 *
 * ```js
 * const input = tf.input({shape: [5]});
 * const denseLayer = tf.layers.dense({units: 1});
 * const activationLayer = tf.layers.activation({activation: 'relu6'});
 *
 * // Obtain the output symbolic tensors by applying the layers in order.
 * const denseOutput = denseLayer.apply(input);
 * const activationOutput = activationLayer.apply(denseOutput);
 *
 * // Create the model based on the inputs.
 * const model = tf.model({
 *     inputs: input,
 *     outputs: [denseOutput, activationOutput]
 * });
 *
 * // Collect both outputs and print separately.
 * const [denseOut, activationOut] = model.predict(tf.randomNormal([6, 5]));
 * denseOut.print();
 * activationOut.print();
 * ```
 *
 * @doc {heading: 'Layers', subheading: 'Basic', namespace: 'layers'}
 */
declare function activation(args: ActivationLayerArgs): Activation;
/**
 * Creates a dense (fully connected) layer.
 *
 * This layer implements the operation:
 *   `output = activation(dot(input, kernel) + bias)`
 *
 * `activation` is the element-wise activation function
 *   passed as the `activation` argument.
 *
 * `kernel` is a weights matrix created by the layer.
 *
 * `bias` is a bias vector created by the layer (only applicable if `useBias`
 * is `true`).
 *
 * **Input shape:**
 *
 *   nD `tf.Tensor` with shape: `(batchSize, ..., inputDim)`.
 *
 *   The most common situation would be
 *   a 2D input with shape `(batchSize, inputDim)`.
 *
 * **Output shape:**
 *
 *   nD tensor with shape: `(batchSize, ..., units)`.
 *
 *   For instance, for a 2D input with shape `(batchSize, inputDim)`,
 *   the output would have shape `(batchSize, units)`.
 *
 * Note: if the input to the layer has a rank greater than 2, then it is
 * flattened prior to the initial dot product with the kernel.
 *
 * @doc {heading: 'Layers', subheading: 'Basic', namespace: 'layers'}
 */
declare function dense(args: DenseLayerArgs): Dense;
/**
 * Applies
 * [dropout](http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf) to
 * the input.
 *
 * Dropout consists in randomly setting a fraction `rate` of input units to 0 at
 * each update during training time, which helps prevent overfitting.
 *
 * @doc {heading: 'Layers', subheading: 'Basic', namespace: 'layers'}
 */
declare function dropout(args: DropoutLayerArgs): Dropout;
/**
 * Spatial 1D version of Dropout.
 *
 * This Layer type performs the same function as the Dropout layer, but it drops
 * entire 1D feature maps instead of individual elements. For example, if an
 * input example consists of 3 timesteps and the feature map for each timestep
 * has a size of 4, a `spatialDropout1d` layer may zero out the feature maps
 * of the 1st timesteps and 2nd timesteps completely while sparing all feature
 * elements of the 3rd timestep.
 *
 * If adjacent frames (timesteps) are strongly correlated (as is normally the
 * case in early convolution layers), regular dropout will not regularize the
 * activation and will otherwise just result in merely an effective learning
 * rate decrease. In this case, `spatialDropout1d` will help promote
 * independence among feature maps and should be used instead.
 *
 * **Arguments:**
 *   rate: A floating-point number >=0 and <=1. Fraction of the input elements
 *     to drop.
 *
 * **Input shape:**
 *   3D tensor with shape `(samples, timesteps, channels)`.
 *
 * **Output shape:**
 *   Same as the input shape.
 *
 * References:
 *   - [Efficient Object Localization Using Convolutional
 *      Networks](https://arxiv.org/abs/1411.4280)
 *
 * @doc {heading: 'Layers', subheading: 'Basic', namespace: 'layers'}
 */
declare function spatialDropout1d(args: SpatialDropout1DLayerConfig): SpatialDropout1D;
/**
 * Flattens the input. Does not affect the batch size.
 *
 * A `Flatten` layer flattens each batch in its inputs to 1D (making the output
 * 2D).
 *
 * For example:
 *
 * ```js
 * const input = tf.input({shape: [4, 3]});
 * const flattenLayer = tf.layers.flatten();
 * // Inspect the inferred output shape of the flatten layer, which
 * // equals `[null, 12]`. The 2nd dimension is 4 * 3, i.e., the result of the
 * // flattening. (The 1st dimension is the undermined batch size.)
 * console.log(JSON.stringify(flattenLayer.apply(input).shape));
 * ```
 *
 * @doc {heading: 'Layers', subheading: 'Basic', namespace: 'layers'}
 */
declare function flatten(args?: FlattenLayerArgs): Flatten;
/**
 * Repeats the input n times in a new dimension.
 *
 * ```js
 *  const model = tf.sequential();
 *  model.add(tf.layers.repeatVector({n: 4, inputShape: [2]}));
 *  const x = tf.tensor2d([[10, 20]]);
 *  // Use the model to do inference on a data point the model hasn't see
 *  model.predict(x).print();
 *  // output shape is now [batch, 2, 4]
 * ```
 *
 * @doc {heading: 'Layers', subheading: 'Basic', namespace: 'layers'}
 */
declare function repeatVector(args: RepeatVectorLayerArgs): RepeatVector;
/**
 * Reshapes an input to a certain shape.
 *
 * ```js
 * const input = tf.input({shape: [4, 3]});
 * const reshapeLayer = tf.layers.reshape({targetShape: [2, 6]});
 * // Inspect the inferred output shape of the Reshape layer, which
 * // equals `[null, 2, 6]`. (The 1st dimension is the undermined batch size.)
 * console.log(JSON.stringify(reshapeLayer.apply(input).shape));
 * ```
 *
 * Input shape:
 *   Arbitrary, although all dimensions in the input shape must be fixed.
 *   Use the configuration `inputShape` when using this layer as the
 *   first layer in a model.
 *
 *
 * Output shape:
 *   [batchSize, targetShape[0], targetShape[1], ...,
 *    targetShape[targetShape.length - 1]].
 *
 * @doc {heading: 'Layers', subheading: 'Basic', namespace: 'layers'}
 */
declare function reshape(args: ReshapeLayerArgs): Reshape;
/**
 * Permutes the dimensions of the input according to a given pattern.
 *
 * Useful for, e.g., connecting RNNs and convnets together.
 *
 * Example:
 *
 * ```js
 * const model = tf.sequential();
 * model.add(tf.layers.permute({
 *   dims: [2, 1],
 *   inputShape: [10, 64]
 * }));
 * console.log(model.outputShape);
 * // Now model's output shape is [null, 64, 10], where null is the
 * // unpermuted sample (batch) dimension.
 * ```
 *
 * Input shape:
 *   Arbitrary. Use the configuration field `inputShape` when using this
 *   layer as the first layer in a model.
 *
 * Output shape:
 *   Same rank as the input shape, but with the dimensions re-ordered (i.e.,
 *   permuted) according to the `dims` configuration of this layer.
 *
 * @doc {heading: 'Layers', subheading: 'Basic', namespace: 'layers'}
 */
declare function permute(args: PermuteLayerArgs): Permute;
/**
 * Maps positive integers (indices) into dense vectors of fixed size.
 * E.g. [[4], [20]] -> [[0.25, 0.1], [0.6, -0.2]]
 *
 * **Input shape:** 2D tensor with shape: `[batchSize, sequenceLength]`.
 *
 * **Output shape:** 3D tensor with shape: `[batchSize, sequenceLength,
 * outputDim]`.
 *
 * @doc {heading: 'Layers', subheading: 'Basic', namespace: 'layers'}
 */
declare function embedding(args: EmbeddingLayerArgs): Embedding;
/**
 * Layer that performs element-wise addition on an `Array` of inputs.
 *
 * It takes as input a list of tensors, all of the same shape, and returns a
 * single tensor (also of the same shape). The inputs are specified as an
 * `Array` when the `apply` method of the `Add` layer instance is called. For
 * example:
 *
 * ```js
 * const input1 = tf.input({shape: [2, 2]});
 * const input2 = tf.input({shape: [2, 2]});
 * const addLayer = tf.layers.add();
 * const sum = addLayer.apply([input1, input2]);
 * console.log(JSON.stringify(sum.shape));
 * // You get [null, 2, 2], with the first dimension as the undetermined batch
 * // dimension.
 * ```
 *
 * @doc {heading: 'Layers', subheading: 'Merge', namespace: 'layers'}
 */
declare function add(args?: LayerArgs): Add;
/**
 * Layer that performs element-wise averaging on an `Array` of inputs.
 *
 * It takes as input a list of tensors, all of the same shape, and returns a
 * single tensor (also of the same shape). For example:
 *
 * ```js
 * const input1 = tf.input({shape: [2, 2]});
 * const input2 = tf.input({shape: [2, 2]});
 * const averageLayer = tf.layers.average();
 * const average = averageLayer.apply([input1, input2]);
 * console.log(JSON.stringify(average.shape));
 * // You get [null, 2, 2], with the first dimension as the undetermined batch
 * // dimension.
 * ```
 *
 * @doc {heading: 'Layers', subheading: 'Merge', namespace: 'layers'}
 */
declare function average(args?: LayerArgs): Average;
/**
 * Layer that concatenates an `Array` of inputs.
 *
 * It takes a list of tensors, all of the same shape except for the
 * concatenation axis, and returns a single tensor, the concatenation
 * of all inputs. For example:
 *
 * ```js
 * const input1 = tf.input({shape: [2, 2]});
 * const input2 = tf.input({shape: [2, 3]});
 * const concatLayer = tf.layers.concatenate();
 * const output = concatLayer.apply([input1, input2]);
 * console.log(JSON.stringify(output.shape));
 * // You get [null, 2, 5], with the first dimension as the undetermined batch
 * // dimension. The last dimension (5) is the result of concatenating the
 * // last dimensions of the inputs (2 and 3).
 * ```
 *
 * @doc {heading: 'Layers', subheading: 'Merge', namespace: 'layers'}
 */
declare function concatenate(args?: ConcatenateLayerArgs): Concatenate;
/**
 * Layer that computes the element-wise maximum of an `Array` of inputs.
 *
 * It takes as input a list of tensors, all of the same shape, and returns a
 * single tensor (also of the same shape). For example:
 *
 * ```js
 * const input1 = tf.input({shape: [2, 2]});
 * const input2 = tf.input({shape: [2, 2]});
 * const maxLayer = tf.layers.maximum();
 * const max = maxLayer.apply([input1, input2]);
 * console.log(JSON.stringify(max.shape));
 * // You get [null, 2, 2], with the first dimension as the undetermined batch
 * // dimension.
 * ```
 *
 * @doc {heading: 'Layers', subheading: 'Merge', namespace: 'layers'}
 */
declare function maximum(args?: LayerArgs): Maximum;
/**
 * Layer that computes the element-wise minimum of an `Array` of inputs.
 *
 * It takes as input a list of tensors, all of the same shape, and returns a
 * single tensor (also of the same shape). For example:
 *
 * ```js
 * const input1 = tf.input({shape: [2, 2]});
 * const input2 = tf.input({shape: [2, 2]});
 * const minLayer = tf.layers.minimum();
 * const min = minLayer.apply([input1, input2]);
 * console.log(JSON.stringify(min.shape));
 * // You get [null, 2, 2], with the first dimension as the undetermined batch
 * // dimension.
 * ```
 *
 * @doc {heading: 'Layers', subheading: 'Merge', namespace: 'layers'}
 */
declare function minimum(args?: LayerArgs): Minimum;
/**
 * Layer that multiplies (element-wise) an `Array` of inputs.
 *
 * It takes as input an Array of tensors, all of the same
 * shape, and returns a single tensor (also of the same shape).
 * For example:
 *
 * ```js
 * const input1 = tf.input({shape: [2, 2]});
 * const input2 = tf.input({shape: [2, 2]});
 * const input3 = tf.input({shape: [2, 2]});
 * const multiplyLayer = tf.layers.multiply();
 * const product = multiplyLayer.apply([input1, input2, input3]);
 * console.log(product.shape);
 * // You get [null, 2, 2], with the first dimension as the undetermined batch
 * // dimension.
 *
 * @doc {heading: 'Layers', subheading: 'Merge', namespace: 'layers'}
 */
declare function multiply(args?: LayerArgs): Multiply;
/**
 * Layer that computes a dot product between samples in two tensors.
 *
 * E.g., if applied to a list of two tensors `a` and `b` both of shape
 * `[batchSize, n]`, the output will be a tensor of shape `[batchSize, 1]`,
 * where each entry at index `[i, 0]` will be the dot product between
 * `a[i, :]` and `b[i, :]`.
 *
 * Example:
 *
 * ```js
 * const dotLayer = tf.layers.dot({axes: -1});
 * const x1 = tf.tensor2d([[10, 20], [30, 40]]);
 * const x2 = tf.tensor2d([[-1, -2], [-3, -4]]);
 *
 * // Invoke the layer's apply() method in eager (imperative) mode.
 * const y = dotLayer.apply([x1, x2]);
 * y.print();
 * ```
 *
 * @doc {heading: 'Layers', subheading: 'Merge', namespace: 'layers'}
 */
declare function dot(args: DotLayerArgs): Dot;
/**
 * Batch normalization layer (Ioffe and Szegedy, 2014).
 *
 * Normalize the activations of the previous layer at each batch,
 * i.e. applies a transformation that maintains the mean activation
 * close to 0 and the activation standard deviation close to 1.
 *
 * Input shape:
 *   Arbitrary. Use the keyword argument `inputShape` (Array of integers, does
 *   not include the sample axis) when calling the constructor of this class,
 *   if this layer is used as a first layer in a model.
 *
 * Output shape:
 *   Same shape as input.
 *
 * References:
 *   - [Batch Normalization: Accelerating Deep Network Training by Reducing
 * Internal Covariate Shift](https://arxiv.org/abs/1502.03167)
 *
 * @doc {heading: 'Layers', subheading: 'Normalization', namespace: 'layers'}
 */
declare function batchNormalization(args?: BatchNormalizationLayerArgs): BatchNormalization;
/**
 * Layer-normalization layer (Ba et al., 2016).
 *
 * Normalizes the activations of the previous layer for each given example in a
 * batch independently, instead of across a batch like in `batchNormalization`.
 * In other words, this layer applies a transformation that maintains the mean
 * activation within each example close to 0 and activation variance close to 1.
 *
 * Input shape:
 *   Arbitrary. Use the argument `inputShape` when using this layer as the first
 *   layer in a model.
 *
 * Output shape:
 *   Same as input.
 *
 * References:
 *   - [Layer Normalization](https://arxiv.org/abs/1607.06450)
 *
 * @doc {heading: 'Layers', subheading: 'Normalization', namespace: 'layers'}
 */
declare function layerNormalization(args?: LayerNormalizationLayerArgs): LayerNormalization;
/**
 * Zero-padding layer for 2D input (e.g., image).
 *
 * This layer can add rows and columns of zeros
 * at the top, bottom, left and right side of an image tensor.
 *
 * Input shape:
 *   4D tensor with shape:
 *   - If `dataFormat` is `"channelsLast"`:
 *     `[batch, rows, cols, channels]`
 *   - If `data_format` is `"channels_first"`:
 *     `[batch, channels, rows, cols]`.
 *
 * Output shape:
 *   4D with shape:
 *   - If `dataFormat` is `"channelsLast"`:
 *     `[batch, paddedRows, paddedCols, channels]`
 *    - If `dataFormat` is `"channelsFirst"`:
 *     `[batch, channels, paddedRows, paddedCols]`.
 *
 * @doc {heading: 'Layers', subheading: 'Padding', namespace: 'layers'}
 */
declare function zeroPadding2d(args?: ZeroPadding2DLayerArgs): ZeroPadding2D;
/**
 * Average pooling operation for spatial data.
 *
 * Input shape: `[batchSize, inLength, channels]`
 *
 * Output shape: `[batchSize, pooledLength, channels]`
 *
 * `tf.avgPool1d` is an alias.
 *
 * @doc {heading: 'Layers', subheading: 'Pooling', namespace: 'layers'}
 */
declare function averagePooling1d(args: Pooling1DLayerArgs): AveragePooling1D;
declare function avgPool1d(args: Pooling1DLayerArgs): AveragePooling1D;
declare function avgPooling1d(args: Pooling1DLayerArgs): AveragePooling1D;
/**
 * Average pooling operation for spatial data.
 *
 * Input shape:
 *  - If `dataFormat === CHANNEL_LAST`:
 *      4D tensor with shape:
 *      `[batchSize, rows, cols, channels]`
 *  - If `dataFormat === CHANNEL_FIRST`:
 *      4D tensor with shape:
 *      `[batchSize, channels, rows, cols]`
 *
 * Output shape
 *  - If `dataFormat === CHANNEL_LAST`:
 *      4D tensor with shape:
 *      `[batchSize, pooledRows, pooledCols, channels]`
 *  - If `dataFormat === CHANNEL_FIRST`:
 *      4D tensor with shape:
 *      `[batchSize, channels, pooledRows, pooledCols]`
 *
 * `tf.avgPool2d` is an alias.
 *
 * @doc {heading: 'Layers', subheading: 'Pooling', namespace: 'layers'}
 */
declare function averagePooling2d(args: Pooling2DLayerArgs): AveragePooling2D;
declare function avgPool2d(args: Pooling2DLayerArgs): AveragePooling2D;
declare function avgPooling2d(args: Pooling2DLayerArgs): AveragePooling2D;
/**
 * Average pooling operation for 3D data.
 *
 * Input shape
 *   - If `dataFormat === channelsLast`:
 *       5D tensor with shape:
 *       `[batchSize, depths, rows, cols, channels]`
 *   - If `dataFormat === channelsFirst`:
 *      4D tensor with shape:
 *       `[batchSize, channels, depths, rows, cols]`
 *
 * Output shape
 *   - If `dataFormat=channelsLast`:
 *       5D tensor with shape:
 *       `[batchSize, pooledDepths, pooledRows, pooledCols, channels]`
 *   - If `dataFormat=channelsFirst`:
 *       5D tensor with shape:
 *       `[batchSize, channels, pooledDepths, pooledRows, pooledCols]`
 *
 * @doc {heading: 'Layers', subheading: 'Pooling', namespace: 'layers'}
 */
declare function averagePooling3d(args: Pooling3DLayerArgs): AveragePooling3D;
declare function avgPool3d(args: Pooling3DLayerArgs): AveragePooling3D;
declare function avgPooling3d(args: Pooling3DLayerArgs): AveragePooling3D;
/**
 * Global average pooling operation for temporal data.
 *
 * Input Shape: 3D tensor with shape: `[batchSize, steps, features]`.
 *
 * Output Shape: 2D tensor with shape: `[batchSize, features]`.
 *
 * @doc {heading: 'Layers', subheading: 'Pooling', namespace: 'layers'}
 */
declare function globalAveragePooling1d(args?: LayerArgs): GlobalAveragePooling1D;
/**
 * Global average pooling operation for spatial data.
 *
 * Input shape:
 *   - If `dataFormat` is `CHANNEL_LAST`:
 *       4D tensor with shape: `[batchSize, rows, cols, channels]`.
 *   - If `dataFormat` is `CHANNEL_FIRST`:
 *       4D tensor with shape: `[batchSize, channels, rows, cols]`.
 *
 * Output shape:
 *   2D tensor with shape: `[batchSize, channels]`.
 *
 * @doc {heading: 'Layers', subheading: 'Pooling', namespace: 'layers'}
 */
declare function globalAveragePooling2d(args: GlobalPooling2DLayerArgs): GlobalAveragePooling2D;
/**
 * Global max pooling operation for temporal data.
 *
 * Input Shape: 3D tensor with shape: `[batchSize, steps, features]`.
 *
 * Output Shape: 2D tensor with shape: `[batchSize, features]`.
 *
 * @doc {heading: 'Layers', subheading: 'Pooling', namespace: 'layers'}
 */
declare function globalMaxPooling1d(args?: LayerArgs): GlobalMaxPooling1D;
/**
 * Global max pooling operation for spatial data.
 *
 * Input shape:
 *   - If `dataFormat` is `CHANNEL_LAST`:
 *       4D tensor with shape: `[batchSize, rows, cols, channels]`.
 *   - If `dataFormat` is `CHANNEL_FIRST`:
 *       4D tensor with shape: `[batchSize, channels, rows, cols]`.
 *
 * Output shape:
 *   2D tensor with shape: `[batchSize, channels]`.
 *
 * @doc {heading: 'Layers', subheading: 'Pooling', namespace: 'layers'}
 */
declare function globalMaxPooling2d(args: GlobalPooling2DLayerArgs): GlobalMaxPooling2D;
/**
 * Max pooling operation for temporal data.
 *
 * Input shape:  `[batchSize, inLength, channels]`
 *
 * Output shape: `[batchSize, pooledLength, channels]`
 *
 * @doc {heading: 'Layers', subheading: 'Pooling', namespace: 'layers'}
 */
declare function maxPooling1d(args: Pooling1DLayerArgs): MaxPooling1D;
/**
 * Max pooling operation for spatial data.
 *
 * Input shape
 *   - If `dataFormat === CHANNEL_LAST`:
 *       4D tensor with shape:
 *       `[batchSize, rows, cols, channels]`
 *   - If `dataFormat === CHANNEL_FIRST`:
 *      4D tensor with shape:
 *       `[batchSize, channels, rows, cols]`
 *
 * Output shape
 *   - If `dataFormat=CHANNEL_LAST`:
 *       4D tensor with shape:
 *       `[batchSize, pooledRows, pooledCols, channels]`
 *   - If `dataFormat=CHANNEL_FIRST`:
 *       4D tensor with shape:
 *       `[batchSize, channels, pooledRows, pooledCols]`
 *
 * @doc {heading: 'Layers', subheading: 'Pooling', namespace: 'layers'}
 */
declare function maxPooling2d(args: Pooling2DLayerArgs): MaxPooling2D;
/**
 * Max pooling operation for 3D data.
 *
 * Input shape
 *   - If `dataFormat === channelsLast`:
 *       5D tensor with shape:
 *       `[batchSize, depths, rows, cols, channels]`
 *   - If `dataFormat === channelsFirst`:
 *      5D tensor with shape:
 *       `[batchSize, channels, depths, rows, cols]`
 *
 * Output shape
 *   - If `dataFormat=channelsLast`:
 *       5D tensor with shape:
 *       `[batchSize, pooledDepths, pooledRows, pooledCols, channels]`
 *   - If `dataFormat=channelsFirst`:
 *       5D tensor with shape:
 *       `[batchSize, channels, pooledDepths, pooledRows, pooledCols]`
 *
 * @doc {heading: 'Layers', subheading: 'Pooling', namespace: 'layers'}
 */
declare function maxPooling3d(args: Pooling3DLayerArgs): MaxPooling3D;
/**
 * Gated Recurrent Unit - Cho et al. 2014.
 *
 * This is an `RNN` layer consisting of one `GRUCell`. However, unlike
 * the underlying `GRUCell`, the `apply` method of `SimpleRNN` operates
 * on a sequence of inputs. The shape of the input (not including the first,
 * batch dimension) needs to be at least 2-D, with the first dimension being
 * time steps. For example:
 *
 * ```js
 * const rnn = tf.layers.gru({units: 8, returnSequences: true});
 *
 * // Create an input with 10 time steps.
 * const input = tf.input({shape: [10, 20]});
 * const output = rnn.apply(input);
 *
 * console.log(JSON.stringify(output.shape));
 * // [null, 10, 8]: 1st dimension is unknown batch size; 2nd dimension is the
 * // same as the sequence length of `input`, due to `returnSequences`: `true`;
 * // 3rd dimension is the `GRUCell`'s number of units.
 *
 * @doc {heading: 'Layers', subheading: 'Recurrent', namespace: 'layers'}
 */
declare function gru(args: GRULayerArgs): GRU;
/**
 * Cell class for `GRU`.
 *
 * `GRUCell` is distinct from the `RNN` subclass `GRU` in that its
 * `apply` method takes the input data of only a single time step and returns
 * the cell's output at the time step, while `GRU` takes the input data
 * over a number of time steps. For example:
 *
 * ```js
 * const cell = tf.layers.gruCell({units: 2});
 * const input = tf.input({shape: [10]});
 * const output = cell.apply(input);
 *
 * console.log(JSON.stringify(output.shape));
 * // [null, 10]: This is the cell's output at a single time step. The 1st
 * // dimension is the unknown batch size.
 * ```
 *
 * Instance(s) of `GRUCell` can be used to construct `RNN` layers. The
 * most typical use of this workflow is to combine a number of cells into a
 * stacked RNN cell (i.e., `StackedRNNCell` internally) and use it to create an
 * RNN. For example:
 *
 * ```js
 * const cells = [
 *   tf.layers.gruCell({units: 4}),
 *   tf.layers.gruCell({units: 8}),
 * ];
 * const rnn = tf.layers.rnn({cell: cells, returnSequences: true});
 *
 * // Create an input with 10 time steps and a length-20 vector at each step.
 * const input = tf.input({shape: [10, 20]});
 * const output = rnn.apply(input);
 *
 * console.log(JSON.stringify(output.shape));
 * // [null, 10, 8]: 1st dimension is unknown batch size; 2nd dimension is the
 * // same as the sequence length of `input`, due to `returnSequences`: `true`;
 * // 3rd dimension is the last `gruCell`'s number of units.
 * ```
 *
 * To create an `RNN` consisting of only *one* `GRUCell`, use the
 * `tf.layers.gru`.
 *
 * @doc {heading: 'Layers', subheading: 'Recurrent', namespace: 'layers'}
 */
declare function gruCell(args: GRUCellLayerArgs): GRUCell;
/**
 * Long-Short Term Memory layer - Hochreiter 1997.
 *
 * This is an `RNN` layer consisting of one `LSTMCell`. However, unlike
 * the underlying `LSTMCell`, the `apply` method of `LSTM` operates
 * on a sequence of inputs. The shape of the input (not including the first,
 * batch dimension) needs to be at least 2-D, with the first dimension being
 * time steps. For example:
 *
 * ```js
 * const lstm = tf.layers.lstm({units: 8, returnSequences: true});
 *
 * // Create an input with 10 time steps.
 * const input = tf.input({shape: [10, 20]});
 * const output = lstm.apply(input);
 *
 * console.log(JSON.stringify(output.shape));
 * // [null, 10, 8]: 1st dimension is unknown batch size; 2nd dimension is the
 * // same as the sequence length of `input`, due to `returnSequences`: `true`;
 * // 3rd dimension is the `LSTMCell`'s number of units.
 *
 * @doc {heading: 'Layers', subheading: 'Recurrent', namespace: 'layers'}
 */
declare function lstm(args: LSTMLayerArgs): LSTM;
/**
 * Cell class for `LSTM`.
 *
 * `LSTMCell` is distinct from the `RNN` subclass `LSTM` in that its
 * `apply` method takes the input data of only a single time step and returns
 * the cell's output at the time step, while `LSTM` takes the input data
 * over a number of time steps. For example:
 *
 * ```js
 * const cell = tf.layers.lstmCell({units: 2});
 * const input = tf.input({shape: [10]});
 * const output = cell.apply(input);
 *
 * console.log(JSON.stringify(output.shape));
 * // [null, 10]: This is the cell's output at a single time step. The 1st
 * // dimension is the unknown batch size.
 * ```
 *
 * Instance(s) of `LSTMCell` can be used to construct `RNN` layers. The
 * most typical use of this workflow is to combine a number of cells into a
 * stacked RNN cell (i.e., `StackedRNNCell` internally) and use it to create an
 * RNN. For example:
 *
 * ```js
 * const cells = [
 *   tf.layers.lstmCell({units: 4}),
 *   tf.layers.lstmCell({units: 8}),
 * ];
 * const rnn = tf.layers.rnn({cell: cells, returnSequences: true});
 *
 * // Create an input with 10 time steps and a length-20 vector at each step.
 * const input = tf.input({shape: [10, 20]});
 * const output = rnn.apply(input);
 *
 * console.log(JSON.stringify(output.shape));
 * // [null, 10, 8]: 1st dimension is unknown batch size; 2nd dimension is the
 * // same as the sequence length of `input`, due to `returnSequences`: `true`;
 * // 3rd dimension is the last `lstmCell`'s number of units.
 * ```
 *
 * To create an `RNN` consisting of only *one* `LSTMCell`, use the
 * `tf.layers.lstm`.
 *
 * @doc {heading: 'Layers', subheading: 'Recurrent', namespace: 'layers'}
 */
declare function lstmCell(args: LSTMCellLayerArgs): LSTMCell;
/**
 * Fully-connected RNN where the output is to be fed back to input.
 *
 * This is an `RNN` layer consisting of one `SimpleRNNCell`. However, unlike
 * the underlying `SimpleRNNCell`, the `apply` method of `SimpleRNN` operates
 * on a sequence of inputs. The shape of the input (not including the first,
 * batch dimension) needs to be at least 2-D, with the first dimension being
 * time steps. For example:
 *
 * ```js
 * const rnn = tf.layers.simpleRNN({units: 8, returnSequences: true});
 *
 * // Create an input with 10 time steps.
 * const input = tf.input({shape: [10, 20]});
 * const output = rnn.apply(input);
 *
 * console.log(JSON.stringify(output.shape));
 * // [null, 10, 8]: 1st dimension is unknown batch size; 2nd dimension is the
 * // same as the sequence length of `input`, due to `returnSequences`: `true`;
 * // 3rd dimension is the `SimpleRNNCell`'s number of units.
 * ```
 *
 * @doc {heading: 'Layers', subheading: 'Recurrent', namespace: 'layers'}
 */
declare function simpleRNN(args: SimpleRNNLayerArgs): SimpleRNN;
/**
 * Cell class for `SimpleRNN`.
 *
 * `SimpleRNNCell` is distinct from the `RNN` subclass `SimpleRNN` in that its
 * `apply` method takes the input data of only a single time step and returns
 * the cell's output at the time step, while `SimpleRNN` takes the input data
 * over a number of time steps. For example:
 *
 * ```js
 * const cell = tf.layers.simpleRNNCell({units: 2});
 * const input = tf.input({shape: [10]});
 * const output = cell.apply(input);
 *
 * console.log(JSON.stringify(output.shape));
 * // [null, 10]: This is the cell's output at a single time step. The 1st
 * // dimension is the unknown batch size.
 * ```
 *
 * Instance(s) of `SimpleRNNCell` can be used to construct `RNN` layers. The
 * most typical use of this workflow is to combine a number of cells into a
 * stacked RNN cell (i.e., `StackedRNNCell` internally) and use it to create an
 * RNN. For example:
 *
 * ```js
 * const cells = [
 *   tf.layers.simpleRNNCell({units: 4}),
 *   tf.layers.simpleRNNCell({units: 8}),
 * ];
 * const rnn = tf.layers.rnn({cell: cells, returnSequences: true});
 *
 * // Create an input with 10 time steps and a length-20 vector at each step.
 * const input = tf.input({shape: [10, 20]});
 * const output = rnn.apply(input);
 *
 * console.log(JSON.stringify(output.shape));
 * // [null, 10, 8]: 1st dimension is unknown batch size; 2nd dimension is the
 * // same as the sequence length of `input`, due to `returnSequences`: `true`;
 * // 3rd dimension is the last `SimpleRNNCell`'s number of units.
 * ```
 *
 * To create an `RNN` consisting of only *one* `SimpleRNNCell`, use the
 * `tf.layers.simpleRNN`.
 *
 * @doc {heading: 'Layers', subheading: 'Recurrent', namespace: 'layers'}
 */
declare function simpleRNNCell(args: SimpleRNNCellLayerArgs): SimpleRNNCell;
/**
 * Convolutional LSTM layer - Xingjian Shi 2015.
 *
 * This is a `ConvRNN2D` layer consisting of one `ConvLSTM2DCell`. However,
 * unlike the underlying `ConvLSTM2DCell`, the `apply` method of `ConvLSTM2D`
 * operates on a sequence of inputs. The shape of the input (not including the
 * first, batch dimension) needs to be 4-D, with the first dimension being time
 * steps. For example:
 *
 * ```js
 * const filters = 3;
 * const kernelSize = 3;
 *
 * const batchSize = 4;
 * const sequenceLength = 2;
 * const size = 5;
 * const channels = 3;
 *
 * const inputShape = [batchSize, sequenceLength, size, size, channels];
 * const input = tf.ones(inputShape);
 *
 * const layer = tf.layers.convLstm2d({filters, kernelSize});
 *
 * const output = layer.apply(input);
 * ```
 */
/** @doc {heading: 'Layers', subheading: 'Recurrent', namespace: 'layers'} */
declare function convLstm2d(args: ConvLSTM2DArgs): ConvLSTM2D;
/**
 * Cell class for `ConvLSTM2D`.
 *
 * `ConvLSTM2DCell` is distinct from the `ConvRNN2D` subclass `ConvLSTM2D` in
 * that its `call` method takes the input data of only a single time step and
 * returns the cell's output at the time step, while `ConvLSTM2D` takes the
 * input data over a number of time steps. For example:
 *
 * ```js
 * const filters = 3;
 * const kernelSize = 3;
 *
 * const sequenceLength = 1;
 * const size = 5;
 * const channels = 3;
 *
 * const inputShape = [sequenceLength, size, size, channels];
 * const input = tf.ones(inputShape);
 *
 * const cell = tf.layers.convLstm2dCell({filters, kernelSize});
 *
 * cell.build(input.shape);
 *
 * const outputSize = size - kernelSize + 1;
 * const outShape = [sequenceLength, outputSize, outputSize, filters];
 *
 * const initialH = tf.zeros(outShape);
 * const initialC = tf.zeros(outShape);
 *
 * const [o, h, c] = cell.call([input, initialH, initialC], {});
 * ```
 */
/** @doc {heading: 'Layers', subheading: 'Recurrent', namespace: 'layers'} */
declare function convLstm2dCell(args: ConvLSTM2DCellArgs): ConvLSTM2DCell;
/**
 * Base class for recurrent layers.
 *
 * Input shape:
 *   3D tensor with shape `[batchSize, timeSteps, inputDim]`.
 *
 * Output shape:
 *   - if `returnState`, an Array of tensors (i.e., `tf.Tensor`s). The first
 *     tensor is the output. The remaining tensors are the states at the
 *     last time step, each with shape `[batchSize, units]`.
 *   - if `returnSequences`, the output will have shape
 *     `[batchSize, timeSteps, units]`.
 *   - else, the output will have shape `[batchSize, units]`.
 *
 * Masking:
 *   This layer supports masking for input data with a variable number
 *   of timesteps. To introduce masks to your data,
 *   use an embedding layer with the `mask_zero` parameter
 *   set to `True`.
 *
 * Notes on using statefulness in RNNs:
 *   You can set RNN layers to be 'stateful', which means that the states
 *   computed for the samples in one batch will be reused as initial states
 *   for the samples in the next batch. This assumes a one-to-one mapping
 *   between samples in different successive batches.
 *
 *   To enable statefulness:
 *     - specify `stateful: true` in the layer constructor.
 *     - specify a fixed batch size for your model, by passing
 *       if sequential model:
 *         `batchInputShape=[...]` to the first layer in your model.
 *       else for functional model with 1 or more Input layers:
 *         `batchShape=[...]` to all the first layers in your model.
 *       This is the expected shape of your inputs *including the batch size*.
 *       It should be a tuple of integers, e.g. `(32, 10, 100)`.
 *     - specify `shuffle=False` when calling fit().
 *
 *   To reset the states of your model, call `.resetStates()` on either
 *   a specific layer, or on your entire model.
 *
 * Note on specifying the initial state of RNNs
 *   You can specify the initial state of RNN layers symbolically by
 *   calling them with the option `initialState`. The value of
 *   `initialState` should be a tensor or list of tensors representing
 *   the initial state of the RNN layer.
 *
 *   You can specify the initial state of RNN layers numerically by
 *   calling `resetStates` with the keyword argument `states`. The value of
 *   `states` should be a numpy array or list of numpy arrays representing
 *   the initial state of the RNN layer.
 *
 * Note on passing external constants to RNNs
 *   You can pass "external" constants to the cell using the `constants`
 *   keyword argument of `RNN.call` method. This requires that the `cell.call`
 *   method accepts the same keyword argument `constants`. Such constants
 *   can be used to condition the cell transformation on additional static
 *   inputs (not changing over time), a.k.a. an attention mechanism.
 *
 * @doc {heading: 'Layers', subheading: 'Recurrent', namespace: 'layers'}
 */
declare function rnn(args: RNNLayerArgs): RNN;
/**
 * Wrapper allowing a stack of RNN cells to behave as a single cell.
 *
 * Used to implement efficient stacked RNNs.
 *
 * @doc {heading: 'Layers', subheading: 'Recurrent', namespace: 'layers'}
 */
declare function stackedRNNCells(args: StackedRNNCellsArgs): StackedRNNCells;
/** @doc {heading: 'Layers', subheading: 'Wrapper', namespace: 'layers'} */
declare function bidirectional(args: BidirectionalLayerArgs): Bidirectional;
/**
 * This wrapper applies a layer to every temporal slice of an input.
 *
 * The input should be at least 3D,  and the dimension of the index `1` will be
 * considered to be the temporal dimension.
 *
 * Consider a batch of 32 samples, where each sample is a sequence of 10 vectors
 * of 16 dimensions. The batch input shape of the layer is then `[32,  10,
 * 16]`, and the `inputShape`, not including the sample dimension, is
 * `[10, 16]`.
 *
 * You can then use `TimeDistributed` to apply a `Dense` layer to each of the 10
 * timesteps, independently:
 *
 * ```js
 * const model = tf.sequential();
 * model.add(tf.layers.timeDistributed({
 *   layer: tf.layers.dense({units: 8}),
 *   inputShape: [10, 16],
 * }));
 *
 * // Now model.outputShape = [null, 10, 8].
 * // The output will then have shape `[32, 10, 8]`.
 *
 * // In subsequent layers, there is no need for `inputShape`:
 * model.add(tf.layers.timeDistributed({layer: tf.layers.dense({units: 32})}));
 * console.log(JSON.stringify(model.outputs[0].shape));
 * // Now model.outputShape = [null, 10, 32].
 * ```
 *
 * The output will then have shape `[32, 10, 32]`.
 *
 * `TimeDistributed` can be used with arbitrary layers, not just `Dense`, for
 * instance a `Conv2D` layer.
 *
 * ```js
 * const model = tf.sequential();
 * model.add(tf.layers.timeDistributed({
 *   layer: tf.layers.conv2d({filters: 64, kernelSize: [3, 3]}),
 *   inputShape: [10, 299, 299, 3],
 * }));
 * console.log(JSON.stringify(model.outputs[0].shape));
 * ```
 *
 * @doc {heading: 'Layers', subheading: 'Wrapper', namespace: 'layers'}
 */
declare function timeDistributed(args: WrapperLayerArgs): TimeDistributed;
declare const globalMaxPool1d: typeof globalMaxPooling1d;
declare const globalMaxPool2d: typeof globalMaxPooling2d;
declare const maxPool1d: typeof maxPooling1d;
declare const maxPool2d: typeof maxPooling2d;
{ Layer, RNN, RNNCell, input };
/**
 * Apply additive zero-centered Gaussian noise.
 *
 * As it is a regularization layer, it is only active at training time.
 *
 * This is useful to mitigate overfitting
 * (you could see it as a form of random data augmentation).
 * Gaussian Noise (GS) is a natural choice as corruption process
 * for real valued inputs.
 *
 * # Arguments
 * stddev: float, standard deviation of the noise distribution.
 *
 * # Input shape
 * Arbitrary. Use the keyword argument `input_shape`
 * (tuple of integers, does not include the samples axis)
 * when using this layer as the first layer in a model.
 *
 * # Output shape
 * Same shape as input.
 *
 * @doc {heading: 'Layers', subheading: 'Noise', namespace: 'layers'}
 */
declare function gaussianNoise(args: GaussianNoiseArgs): GaussianNoise;
/**
 * Apply multiplicative 1-centered Gaussian noise.
 *
 * As it is a regularization layer, it is only active at training time.
 *
 * Arguments:
 *   - `rate`: float, drop probability (as with `Dropout`).
 *     The multiplicative noise will have
 *     standard deviation `sqrt(rate / (1 - rate))`.
 *
 * Input shape:
 *   Arbitrary. Use the keyword argument `inputShape`
 *   (tuple of integers, does not include the samples axis)
 *   when using this layer as the first layer in a model.
 *
 * Output shape:
 *   Same shape as input.
 *
 * References:
 *   - [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](
 *      http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)
 *
 * @doc {heading: 'Layers', subheading: 'Noise', namespace: 'layers'}
 */
declare function gaussianDropout(args: GaussianDropoutArgs): GaussianDropout;
/**
 * Applies Alpha Dropout to the input.
 *
 * As it is a regularization layer, it is only active at training time.
 *
 * Alpha Dropout is a `Dropout` that keeps mean and variance of inputs
 * to their original values, in order to ensure the self-normalizing property
 * even after this dropout.
 * Alpha Dropout fits well to Scaled Exponential Linear Units
 * by randomly setting activations to the negative saturation value.
 *
 * Arguments:
 *   - `rate`: float, drop probability (as with `Dropout`).
 *     The multiplicative noise will have
 *     standard deviation `sqrt(rate / (1 - rate))`.
 *   - `noise_shape`: A 1-D `Tensor` of type `int32`, representing the
 *     shape for randomly generated keep/drop flags.
 *
 * Input shape:
 *   Arbitrary. Use the keyword argument `inputShape`
 *   (tuple of integers, does not include the samples axis)
 *   when using this layer as the first layer in a model.
 *
 * Output shape:
 *   Same shape as input.
 *
 * References:
 *   - [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)
 *
 * @doc {heading: 'Layers', subheading: 'Noise', namespace: 'layers'}
 */
declare function alphaDropout(args: AlphaDropoutArgs): AlphaDropout;
/**
 * Masks a sequence by using a mask value to skip timesteps.
 *
 * If all features for a given sample timestep are equal to `mask_value`,
 * then the sample timestep will be masked (skipped) in all downstream layers
 * (as long as they support masking).
 *
 * If any downstream layer does not support masking yet receives such
 * an input mask, an exception will be raised.
 *
 * Arguments:
 *   - `maskValue`: Either None or mask value to skip.
 *
 * Input shape:
 *   Arbitrary. Use the keyword argument `inputShape`
 *   (tuple of integers, does not include the samples axis)
 *   when using this layer as the first layer in a model.
 *
 * Output shape:
 *   Same shape as input.
 *
 * @doc {heading: 'Layers', subheading: 'Mask', namespace: 'layers'}
 */
declare function masking(args?: MaskingArgs): Masking;
/**
 * A preprocessing layer which rescales input values to a new range.
 *
 * This layer rescales every value of an input (often an image) by multiplying
 * by `scale` and adding `offset`.
 *
 * For instance:
 * 1. To rescale an input in the ``[0, 255]`` range
 * to be in the `[0, 1]` range, you would pass `scale=1/255`.
 * 2. To rescale an input in the ``[0, 255]`` range to be in the `[-1, 1]`
 * range, you would pass `scale=1./127.5, offset=-1`.
 * The rescaling is applied both during training and inference. Inputs can be
 * of integer or floating point dtype, and by default the layer will output
 * floats.
 *
 * Arguments:
 *   - `scale`: Float, the scale to apply to the inputs.
 *   - `offset`: Float, the offset to apply to the inputs.
 *
 * Input shape:
 *   Arbitrary.
 *
 * Output shape:
 *   Same as input.
 *
 * @doc {heading: 'Layers', subheading: 'Rescaling', namespace: 'layers'}
 */
declare function rescaling(args?: RescalingArgs): Rescaling;
/**
 *  A preprocessing layer which center crops images.
 *
 *   This layers crops the central portion of the images to a target size. If an
 *   image is smaller than the target size, it will be resized and cropped so as
 *   to return the largest possible window in the image that matches the target
 *   aspect ratio.
 *
 *   Input pixel values can be of any range (e.g. `[0., 1.)` or `[0, 255]`) and
 *   of integer or floating point dtype.
 *
 *   If the input height/width is even and the target height/width is odd (or
 *   inversely), the input image is left-padded by 1 pixel.
 *
 *   Arguments:
 *     `height`: Integer, the height of the output shape.
 *     `width`: Integer, the width of the output shape.
 *
 *   Input shape:
 *     3D (unbatched) or 4D (batched) tensor with shape:
 *     `(..., height, width, channels)`, in `channelsLast` format.
 *
 *   Output shape:
 *     3D (unbatched) or 4D (batched) tensor with shape:
 *     `(..., targetHeight, targetWidth, channels)`.
 *
 *
 *  @doc {heading: 'Layers', subheading: 'CenterCrop', namespace: 'layers'}
 */
declare function centerCrop(args?: CenterCropArgs): CenterCrop;
/**
 * A preprocessing layer which resizes images.
 * This layer resizes an image input to a target height and width. The input
 * should be a 4D (batched) or 3D (unbatched) tensor in `"channels_last"`
 * format.  Input pixel values can be of any range (e.g. `[0., 1.)` or `[0,
 * 255]`) and of interger or floating point dtype. By default, the layer will
 * output floats.
 *
 * Arguments:
 *   - `height`: number, the height for the output tensor.
 *   - `width`: number, the width for the output tensor.
 *   - `interpolation`: string, the method for image resizing interpolation.
 *   - `cropToAspectRatio`: boolean, whether to keep image aspect ratio.
 *
 * Input shape:
 *   Arbitrary.
 *
 * Output shape:
 *   height, width, num channels.
 *
 * @doc {heading: 'Layers', subheading: 'Resizing', namespace: 'layers'}
 */
declare function resizing(args?: ResizingArgs): Resizing;
/**
 * A preprocessing layer which encodes integer features.
 *
 * This layer provides options for condensing data into a categorical encoding
 * when the total number of tokens are known in advance. It accepts integer
 * values as inputs, and it outputs a dense representation of those
 * inputs.
 *
 * Arguments:
 *
 * numTokens: The total number of tokens the layer should support. All
 *  inputs to the layer must integers in the range `0 <= value <
 *  numTokens`, or an error will be thrown.
 *
 * outputMode: Specification for the output of the layer.
 *  Defaults to `multiHot`. Values can be `oneHot`, `multiHot` or
 *  `count`, configuring the layer as follows:
 *
 *    oneHot: Encodes each individual element in the input into an
 *      array of `numTokens` size, containing a 1 at the element index. If
 *      the last dimension is size 1, will encode on that dimension. If the
 *      last dimension is not size 1, will append a new dimension for the
 *      encoded output.
 *
 *    multiHot: Encodes each sample in the input into a single array
 *     of `numTokens` size, containing a 1 for each vocabulary term
 *     present in the sample. Treats the last dimension as the sample
 *     dimension, if input shape is `(..., sampleLength)`, output shape
 *     will be `(..., numTokens)`.
 *
 *    count: Like `multiHot`, but the int array contains a count of
 *     the number of times the token at that index appeared in the sample.
 *
 *  For all output modes, currently only output up to rank 2 is supported.
 *   Call arguments:
 *    inputs: A 1D or 2D tensor of integer inputs.
 *    countWeights: A tensor in the same shape as `inputs` indicating the
 *    weight for each sample value when summing up in `count` mode. Not used
 *    in `multiHot` or `oneHot` modes.
 *
 *
 * @doc {heading: 'Layers', subheading: 'CategoryEncoding', namespace: 'layers'}
 */
declare function categoryEncoding(args: CategoryEncodingArgs): CategoryEncoding;
/// <amd-module name="@tensorflow/tfjs-layers/dist/exports_metrics" />

/**
 * Binary accuracy metric function.
 *
 * `yTrue` and `yPred` can have 0-1 values. Example:
 * ```js
 * const x = tf.tensor2d([[1, 1, 1, 1], [0, 0, 0, 0]], [2, 4]);
 * const y = tf.tensor2d([[1, 0, 1, 0], [0, 0, 0, 1]], [2, 4]);
 * const accuracy = tf.metrics.binaryAccuracy(x, y);
 * accuracy.print();
 * ```
 *
 * `yTrue` and `yPred` can also have floating-number values between 0 and 1, in
 * which case the values will be thresholded at 0.5 to yield 0-1 values (i.e.,
 * a value >= 0.5 and <= 1.0 is interpreted as 1).
 *
 * Example:
 * ```js
 * const x = tf.tensor1d([1, 1, 1, 1, 0, 0, 0, 0]);
 * const y = tf.tensor1d([0.2, 0.4, 0.6, 0.8, 0.2, 0.3, 0.4, 0.7]);
 * const accuracy = tf.metrics.binaryAccuracy(x, y);
 * accuracy.print();
 * ```
 *
 * @param yTrue Binary Tensor of truth.
 * @param yPred Binary Tensor of prediction.
 * @return Accuracy Tensor.
 *
 * @doc {heading: 'Metrics', namespace: 'metrics'}
 */
declare function binaryAccuracy(yTrue: Tensor, yPred: Tensor): Tensor;
/**
 * Binary crossentropy metric function.
 *
 * Example:
 * ```js
 * const x = tf.tensor2d([[0], [1], [1], [1]]);
 * const y = tf.tensor2d([[0], [0], [0.5], [1]]);
 * const crossentropy = tf.metrics.binaryCrossentropy(x, y);
 * crossentropy.print();
 * ```
 *
 * @param yTrue Binary Tensor of truth.
 * @param yPred Binary Tensor of prediction, probabilities for the `1` case.
 * @return Accuracy Tensor.
 *
 * @doc {heading: 'Metrics', namespace: 'metrics'}
 */
declare function binaryCrossentropy(yTrue: Tensor, yPred: Tensor): Tensor;
/**
 * Sparse categorical accuracy metric function.
 *
 * Example:
 * ```js
 *
 * const yTrue = tf.tensor1d([1, 1, 2, 2, 0]);
 * const yPred = tf.tensor2d(
 *      [[0, 1, 0], [1, 0, 0], [0, 0.4, 0.6], [0, 0.6, 0.4], [0.7, 0.3, 0]]);
 * const crossentropy = tf.metrics.sparseCategoricalAccuracy(yTrue, yPred);
 * crossentropy.print();
 * ```
 *
 * @param yTrue True labels: indices.
 * @param yPred Predicted probabilities or logits.
 * @returns Accuracy tensor.
 *
 * @doc {heading: 'Metrics', namespace: 'metrics'}
 */
declare function sparseCategoricalAccuracy(yTrue: Tensor, yPred: Tensor): Tensor;
/**
 * Categorical accuracy metric function.
 *
 * Example:
 * ```js
 * const x = tf.tensor2d([[0, 0, 0, 1], [0, 0, 0, 1]]);
 * const y = tf.tensor2d([[0.1, 0.8, 0.05, 0.05], [0.1, 0.05, 0.05, 0.8]]);
 * const accuracy = tf.metrics.categoricalAccuracy(x, y);
 * accuracy.print();
 * ```
 *
 * @param yTrue Binary Tensor of truth: one-hot encoding of categories.
 * @param yPred Binary Tensor of prediction: probabilities or logits for the
 *   same categories as in `yTrue`.
 * @return Accuracy Tensor.
 *
 * @doc {heading: 'Metrics', namespace: 'metrics'}
 */
declare function categoricalAccuracy(yTrue: Tensor, yPred: Tensor): Tensor;
/**
 * Categorical crossentropy between an output tensor and a target tensor.
 *
 * @param target A tensor of the same shape as `output`.
 * @param output A tensor resulting from a softmax (unless `fromLogits` is
 *  `true`, in which case `output` is expected to be the logits).
 * @param fromLogits Boolean, whether `output` is the result of a softmax, or is
 *   a tensor of logits.
 *
 * @doc {heading: 'Metrics', namespace: 'metrics'}
 */
declare function categoricalCrossentropy(yTrue: Tensor, yPred: Tensor): Tensor;
/**
 * Computes the precision of the predictions with respect to the labels.
 *
 * Example:
 * ```js
 * const x = tf.tensor2d(
 *    [
 *      [0, 0, 0, 1],
 *      [0, 1, 0, 0],
 *      [0, 0, 0, 1],
 *      [1, 0, 0, 0],
 *      [0, 0, 1, 0]
 *    ]
 * );
 *
 * const y = tf.tensor2d(
 *    [
 *      [0, 0, 1, 0],
 *      [0, 1, 0, 0],
 *      [0, 0, 0, 1],
 *      [0, 1, 0, 0],
 *      [0, 1, 0, 0]
 *    ]
 * );
 *
 * const precision = tf.metrics.precision(x, y);
 * precision.print();
 * ```
 *
 * @param yTrue The ground truth values. Expected to contain only 0-1 values.
 * @param yPred The predicted values. Expected to contain only 0-1 values.
 * @return Precision Tensor.
 *
 * @doc {heading: 'Metrics', namespace: 'metrics'}
 */
declare function precision(yTrue: Tensor, yPred: Tensor): Tensor;
/**
 * Computes the recall of the predictions with respect to the labels.
 *
 * Example:
 * ```js
 * const x = tf.tensor2d(
 *    [
 *      [0, 0, 0, 1],
 *      [0, 1, 0, 0],
 *      [0, 0, 0, 1],
 *      [1, 0, 0, 0],
 *      [0, 0, 1, 0]
 *    ]
 * );
 *
 * const y = tf.tensor2d(
 *    [
 *      [0, 0, 1, 0],
 *      [0, 1, 0, 0],
 *      [0, 0, 0, 1],
 *      [0, 1, 0, 0],
 *      [0, 1, 0, 0]
 *    ]
 * );
 *
 * const recall = tf.metrics.recall(x, y);
 * recall.print();
 * ```
 *
 * @param yTrue The ground truth values. Expected to contain only 0-1 values.
 * @param yPred The predicted values. Expected to contain only 0-1 values.
 * @return Recall Tensor.
 *
 * @doc {heading: 'Metrics', namespace: 'metrics'}
 */
declare function recall(yTrue: Tensor, yPred: Tensor): Tensor;
/**
 * Loss or metric function: Cosine proximity.
 *
 * Mathematically, cosine proximity is defined as:
 *   `-sum(l2Normalize(yTrue) * l2Normalize(yPred))`,
 * wherein `l2Normalize()` normalizes the L2 norm of the input to 1 and `*`
 * represents element-wise multiplication.
 *
 * ```js
 * const yTrue = tf.tensor2d([[1, 0], [1, 0]]);
 * const yPred = tf.tensor2d([[1 / Math.sqrt(2), 1 / Math.sqrt(2)], [0, 1]]);
 * const proximity = tf.metrics.cosineProximity(yTrue, yPred);
 * proximity.print();
 * ```
 *
 * @param yTrue Truth Tensor.
 * @param yPred Prediction Tensor.
 * @return Cosine proximity Tensor.
 *
 * @doc {heading: 'Metrics', namespace: 'metrics'}
 */
declare function cosineProximity(yTrue: Tensor, yPred: Tensor): Tensor;
/**
 * Loss or metric function: Mean absolute error.
 *
 * Mathematically, mean absolute error is defined as:
 *   `mean(abs(yPred - yTrue))`,
 * wherein the `mean` is applied over feature dimensions.
 *
 * ```js
 * const yTrue = tf.tensor2d([[0, 1], [0, 0], [2, 3]]);
 * const yPred = tf.tensor2d([[0, 1], [0, 1], [-2, -3]]);
 * const mse = tf.metrics.meanAbsoluteError(yTrue, yPred);
 * mse.print();
 * ```
 *
 * @param yTrue Truth Tensor.
 * @param yPred Prediction Tensor.
 * @return Mean absolute error Tensor.
 *
 * @doc {heading: 'Metrics', namespace: 'metrics'}
 */
declare function meanAbsoluteError(yTrue: Tensor, yPred: Tensor): Tensor;
/**
 * Loss or metric function: Mean absolute percentage error.
 *
 * ```js
 * const yTrue = tf.tensor2d([[0, 1], [10, 20]]);
 * const yPred = tf.tensor2d([[0, 1], [11, 24]]);
 * const mse = tf.metrics.meanAbsolutePercentageError(yTrue, yPred);
 * mse.print();
 * ```
 *
 * Aliases: `tf.metrics.MAPE`, `tf.metrics.mape`.
 *
 * @param yTrue Truth Tensor.
 * @param yPred Prediction Tensor.
 * @return Mean absolute percentage error Tensor.
 *
 * @doc {heading: 'Metrics', namespace: 'metrics'}
 */
declare function meanAbsolutePercentageError(yTrue: Tensor, yPred: Tensor): Tensor;
declare function MAPE(yTrue: Tensor, yPred: Tensor): Tensor;
declare function mape(yTrue: Tensor, yPred: Tensor): Tensor;
/**
 * Loss or metric function: Mean squared error.
 *
 * ```js
 * const yTrue = tf.tensor2d([[0, 1], [3, 4]]);
 * const yPred = tf.tensor2d([[0, 1], [-3, -4]]);
 * const mse = tf.metrics.meanSquaredError(yTrue, yPred);
 * mse.print();
 * ```
 *
 * Aliases: `tf.metrics.MSE`, `tf.metrics.mse`.
 *
 * @param yTrue Truth Tensor.
 * @param yPred Prediction Tensor.
 * @return Mean squared error Tensor.
 *
 * @doc {heading: 'Metrics', namespace: 'metrics'}
 */
declare function meanSquaredError(yTrue: Tensor, yPred: Tensor): Tensor;
declare function MSE(yTrue: Tensor, yPred: Tensor): Tensor;
declare function mse(yTrue: Tensor, yPred: Tensor): Tensor;

/// <amd-module name="@tensorflow/tfjs-layers/dist/exports_models" />
/// <amd-module name="@tensorflow/tfjs-layers/dist/exports_regularizers" />
/**
 * Regularizer for L1 and L2 regularization.
 *
 * Adds a term to the loss to penalize large weights:
 * loss += sum(l1 * abs(x)) + sum(l2 * x^2)
 *
 * @doc {heading: 'Regularizers', namespace: 'regularizers'}
 */
declare function l1l2(config?: L1L2Args): Regularizer;
/**
 * Regularizer for L1 regularization.
 *
 * Adds a term to the loss to penalize large weights:
 * loss += sum(l1 * abs(x))
 * @param args l1 config.
 *
 * @doc {heading: 'Regularizers', namespace: 'regularizers'}
 */
declare function l1(config?: L1Args): Regularizer;
/**
 * Regularizer for L2 regularization.
 *
 * Adds a term to the loss to penalize large weights:
 * loss += sum(l2 * x^2)
 * @param args l2 config.
 *
 * @doc {heading: 'Regularizers', namespace: 'regularizers'}
 */
declare function l2(config?: L2Args): Regularizer;

/// <amd-module name="@tensorflow/tfjs-core/dist/gradients/Exp_grad" />
declare const expGradConfig: GradConfig;

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/exp_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/eye" />
/**
 * Create an identity matrix.
 *
 * @param numRows Number of rows.
 * @param numColumns Number of columns. Defaults to `numRows`.
 * @param batchShape If provided, will add the batch shape to the beginning
 *   of the shape of the returned `tf.Tensor` by repeating the identity
 *   matrix.
 * @param dtype Data type.
 * @returns Identity matrix of the specified size and data type, possibly
 *   with batch repetition if `batchShape` is specified.
 *
 * @doc {heading: 'Tensors', subheading: 'Creation'}
 */
declare function eye_(numRows: number, numColumns?: number, batchShape?: [
    number
] | [
    number,
    number
] | [number, number, number] | [number, number, number, number], dtype?: DataType): Tensor2D;
declare const eye: typeof eye_;
{ };
/// <amd-module name="@tensorflow/tfjs-core/dist/ops/eye_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/public/chained_ops/fft" />
declare module '../../tensor' {
    interface Tensor<R extends Rank = Rank> {
        fft<T extends Tensor>(this: Tensor): Tensor;
    }
}

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/spectral/fft_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-data/dist/iterators/file_chunk_iterator" />
interface FileChunkIteratorOptions {
    /** The byte offset at which to begin reading the File or Blob. Default 0. */
    offset?: number;
    /** The number of bytes to read at a time. Default 1MB. */
    chunkSize?: number;
}
/**
 * Provide a stream of chunks from a File, Blob, or Uint8Array.
 * @param file The source File, Blob or Uint8Array.
 * @param options Optional settings controlling file reading.
 * @returns a lazy Iterator of Uint8Arrays containing sequential chunks of the
 *   input File, Blob or Uint8Array.
 */
declare class FileChunkIterator extends ByteChunkIterator {
    protected file: FileElement;
    protected options: FileChunkIteratorOptions;
    offset: number;
    chunkSize: number;
    constructor(file: FileElement, options?: FileChunkIteratorOptions);
    summary(): string;
    next(): Promise<IteratorResult<Uint8Array>>;
}

/// <amd-module name="@tensorflow/tfjs-data/dist/sources/file_data_source" />
/**
 * Represents a file, blob, or Uint8Array readable as a stream of binary data
 * chunks.
 */
declare class FileDataSource extends DataSource {
    protected input: FileElement | string;
    protected readonly options: FileChunkIteratorOptions;
    /**
     * Create a `FileDataSource`.
     *
     * @param input Local file path, or `File`/`Blob`/`Uint8Array` object to
     *     read. Local file only works in node environment.
     * @param options Options passed to the underlying `FileChunkIterator`s,
     *   such as {chunksize: 1024}.
     */
    constructor(input: FileElement | string, options?: FileChunkIteratorOptions);
    iterator(): Promise<ByteChunkIterator>;
}

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/fill" />
/**
 * Creates a `tf.Tensor` filled with a scalar value.
 *
 * ```js
 * tf.fill([2, 2], 4).print();
 * ```
 *
 * @param shape An array of integers defining the output tensor shape.
 * @param value The scalar value to fill the tensor with.
 * @param dtype The type of an element in the resulting tensor. Defaults to
 * 'float'.
 *
 * @doc {heading: 'Tensors', subheading: 'Creation'}
 */
declare function fill<R extends Rank>(shape: ShapeMap[R], value: number | string, dtype?: DataType): Tensor<R>;
{ fill };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/fill_test" />
{ };
/// <amd-module name="@tensorflow/tfjs-core/dist/flags" />


/// <amd-module name="@tensorflow/tfjs-layers/dist/flags_layers" />
declare const ENV: import("@tensorflow/tfjs-core").Environment;

/// <amd-module name="@tensorflow/tfjs-core/dist/flags_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/public/chained_ops/flatten" />
declare module '../../tensor' {
    interface Tensor<R extends Rank = Rank> {
        flatten<T extends Tensor>(): Tensor1D;
    }
}

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/image/flip_left_right" />
/**
 * Flips the image left to right. Currently available in the CPU, WebGL, and
 * WASM backends.
 *
 * @param image 4d tensor of shape `[batch, imageHeight, imageWidth, depth]`.
 */
/** @doc {heading: 'Operations', subheading: 'Images', namespace: 'image'} */
declare function flipLeftRight_(image: Tensor4D | TensorLike): Tensor4D;
declare const flipLeftRight: typeof flipLeftRight_;
{ };
/// <amd-module name="@tensorflow/tfjs-core/dist/ops/image/flip_left_right_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/public/chained_ops/floor" />
declare module '../../tensor' {
    interface Tensor<R extends Rank = Rank> {
        floor<T extends Tensor>(this: T): T;
    }
}
/// <amd-module name="@tensorflow/tfjs-core/dist/public/chained_ops/floorDiv" />
declare module '../../tensor' {
    interface Tensor<R extends Rank = Rank> {
        floorDiv<T extends Tensor>(b: Tensor | TensorLike): T;
    }
}

/// <amd-module name="@tensorflow/tfjs-core/dist/gradients/FloorDiv_grad" />
declare const floorDivGradConfig: GradConfig;

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/floordiv_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/gradients/Floor_grad" />
declare const floorGradConfig: GradConfig;

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/floor_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/signal/frame" />
/**
 * Expands input into frames of frameLength.
 * Slides a window size with frameStep.
 *
 * ```js
 * tf.signal.frame([1, 2, 3], 2, 1).print();
 * ```
 * @param signal The input tensor to be expanded
 * @param frameLength Length of each frame
 * @param frameStep The frame hop size in samples.
 * @param padEnd Whether to pad the end of signal with padValue.
 * @param padValue A number to use where the input signal does
 *     not exist when padEnd is True.
 *
 * @doc {heading: 'Operations', subheading: 'Signal', namespace: 'signal'}
 */
declare function frame_(signal: Tensor1D, frameLength: number, frameStep: number, padEnd?: boolean, padValue?: number): Tensor;
declare const frame: typeof frame_;
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/signal/frame_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/from_pixels_test" />
{ };
/// <amd-module name="@tensorflow/tfjs-core/dist/gradients/FusedBatchNorm_grad" />
declare const fusedBatchNormGradConfig: GradConfig;

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/fused/fused_conv2d_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/fused/fused_depthwise_conv2d_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/fused/fused_mat_mul_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/fused_ops" />
{ Activation, conv2d, depthwiseConv2d, matMul };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/fused_types" />
declare type FusedConv2DConfig = {
    input: Tensor4D;
    filter: Tensor4D;
    convInfo: Conv2DInfo;
    bias?: Tensor;
    activation?: Activation;
    preluActivationWeights?: Tensor;
    leakyreluAlpha?: number;
};
declare type FusedBatchMatMulConfig = {
    a: Tensor3D;
    b: Tensor3D;
    transposeA: boolean;
    transposeB: boolean;
    bias?: Tensor;
    activation?: Activation;
    preluActivationWeights?: Tensor;
    leakyreluAlpha?: number;
};
declare type Activation = 'linear' | 'relu' | 'prelu' | 'elu' | 'relu6' | 'leakyrelu' | 'sigmoid';

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/fused_util" />
declare function getFusedDyActivation(dy: Tensor, y: Tensor, activation: Activation): Tensor;
declare function getFusedBiasGradient(bias: Tensor, dyActivation: Tensor): Tensor;
declare function applyActivation(x: Tensor, activation: Activation, preluActivationWeights?: Tensor, leakyreluAlpha?: number): Tensor;
declare const shouldFuse: (gradientDepth: number, activation: Activation) => boolean;

/// <amd-module name="@tensorflow/tfjs-core/dist/public/chained_ops/gather" />
declare module '../../tensor' {
    interface Tensor<R extends Rank = Rank> {
        gather<T extends Tensor>(this: T, indices: Tensor | TensorLike, axis?: number): T;
    }
}

/// <amd-module name="@tensorflow/tfjs-core/dist/gradients/GatherV2_grad" />
declare const gatherGradConfig: GradConfig;
/// <amd-module name="@tensorflow/tfjs-core/dist/ops/gather_nd" />
/**
 * Gather slices from input tensor into a Tensor with shape specified by
 * `indices`.
 *
 * `indices` is a K-dimensional integer tensor, best thought of as a
 * (K-1)-dimensional tensor of indices into input, where each element defines a
 * slice of input:
 * output[\\(i_0, ..., i_{K-2}\\)] = input[indices[\\(i_0, ..., i_{K-2}\\)]]
 *
 * Whereas in `tf.gather`, `indices` defines slices into the first dimension of
 * input, in `tf.gatherND`, `indices` defines slices into the first N dimensions
 * of input, where N = indices.shape[-1].
 *
 * The last dimension of indices can be at most the rank of input:
 * indices.shape[-1] <= input.rank
 *
 * The last dimension of `indices` corresponds to elements
 * (if indices.shape[-1] == input.rank) or slices
 * (if indices.shape[-1] < input.rank) along dimension indices.shape[-1] of
 * input.
 * The output tensor has shape
 * indices.shape[:-1] + input.shape[indices.shape[-1]:]
 *
 * Note that on CPU, if an out of bound index is found, an error is returned. On
 * GPU, if an out of bound index is found, a 0 is stored in the corresponding
 * output value.
 *
 * ```js
 * const indices = tf.tensor2d([0, 1, 1, 0], [2,2], 'int32');
 * const input = tf.tensor2d([9, 10, 11, 12], [2, 2]);
 * tf.gatherND(input, indices).print() // [10, 11]
 * ```
 *
 * @param x The tensor from which to gather values.
 * @param indices Index tensor, must be of type int32.
 *
 * @doc {heading: 'Operations', subheading: 'Slicing and Joining'}
 */
declare function gatherND_(x: Tensor | TensorLike, indices: Tensor | TensorLike): Tensor;
declare const gatherND: typeof gatherND_;
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/gather_nd_test" />
{ };
/// <amd-module name="@tensorflow/tfjs-core/dist/ops/gather_nd_util" />

/**
 * Validate gather nd inputs.
 *
 * @param tensor The tensor contains the source values.
 * @param indices The tensor contains the indices to slice the source.
 *
 * @returns [resultShape, numUpdates, sliceSize, strides]
 */
declare function prepareAndValidate(tensor: TensorInfo, indices: TensorInfo): [
    number[],
    number,
    number,
    number[]
];

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/gather_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-layers/dist/utils/generic_utils" />
/**
 * If `value` is an Array, equivalent to Python's `value * numValues`.
 * If `value` is not an Array, equivalent to Python's `[value] * numValues`
 */
declare function pyListRepeat(value: any, numValues: number): any[];
declare function assert(val: boolean, message?: string): void;
/**
 * Count the number of elements of the `array` that are equal to `reference`.
 */
declare function count<T>(array: T[], refernce: T): number;
/**
 * If an array is of length 1, just return the first element. Otherwise, return
 * the full array.
 * @param tensors
 */
declare function singletonOrArray<T>(xs: T[]): T | T[];
/**
 * Normalizes a list/tensor into a list.
 *
 * If a tensor is passed, we return
 * a list of size 1 containing the tensor.
 *
 * @param x target object to be normalized.
 */
declare function toList(x: any): any[];
/**
 * Generate a UID for a list
 */
declare function objectListUid(objs: any | any[]): string;
/**
 * Converts string to snake-case.
 * @param name
 */
declare function toSnakeCase(name: string): string;
declare function toCamelCase(identifier: string): string;
declare function serializeKerasObject(instance: serialization.Serializable): serialization.ConfigDictValue;
/**
 * Deserialize a saved Keras Object
 * @param identifier either a string ID or a saved Keras dictionary
 * @param moduleObjects a list of Python class names to object constructors
 * @param customObjects a list of Python class names to object constructors
 * @param printableModuleName debug text for the object being reconstituted
 * @param fastWeightInit Optional flag to use fast weight initialization
 *   during deserialization. This is applicable to cases in which
 *   the initialization will be immediately overwritten by loaded weight
 *   values. Default: `false`.
 * @returns a TensorFlow.js Layers object
 */
declare function deserializeKerasObject(identifier: string | serialization.ConfigDict, moduleObjects?: {
    [objName: string]: any;
}, customObjects?: {
    [objName: string]: any;
}, printableModuleName?: string, fastWeightInit?: boolean): any;
/**
 * Compares two numbers for sorting.
 * @param a
 * @param b
 */
declare function numberCompare(a: number, b: number): 0 | 1 | -1;
/**
 * Comparison of two numbers for reverse sorting.
 * @param a
 * @param b
 */
declare function reverseNumberCompare(a: number, b: number): number;
/**
 * Convert a string into the corresponding DType.
 * @param dtype
 * @returns An instance of DType.
 */
declare function stringToDType(dtype: string): DataType;
/**
 * Test the element-by-element equality of two Arrays of strings.
 * @param xs First array of strings.
 * @param ys Second array of strings.
 * @returns Wether the two arrays are all equal, element by element.
 */
declare function stringsEqual(xs: string[], ys: string[]): boolean;
/**
 * Get the unique elements of an array.
 * @param xs Array.
 * @returns An Array consisting of the unique elements in `xs`.
 */
declare function unique<T>(xs: T[]): T[];
/**
 * Determine if an Object is empty (i.e., does not have own properties).
 * @param obj Object
 * @returns Whether the Object is empty.
 * @throws ValueError: If object is `null` or `undefined`.
 */
declare function isObjectEmpty(obj: {}): boolean;
/**
 * Helper function used to build type union/enum run-time checkers.
 * @param values The list of allowed values.
 * @param label A string name for the type
 * @param value The value to test.
 * @throws ValueError: If the value is not in values nor `undefined`/`null`.
 */
declare function checkStringTypeUnionValue(values: string[], label: string, value: string): void;
/**
 * Helper function for verifying the types of inputs.
 *
 * Ensures that the elements of `x` are all of type `expectedType`.
 * Also verifies that the length of `x` is within bounds.
 *
 * @param x Object to test.
 * @param expectedType The string expected type of all of the elements in the
 * Array.
 * @param minLength Return false if x.length is less than this.
 * @param maxLength Return false if x.length is greater than this.
 * @returns true if and only if `x` is an `Array<expectedType>` with
 * length >= `minLength` and <= `maxLength`.
 */
declare function checkArrayTypeAndLength(x: any, expectedType: string, minLength?: number, maxLength?: number): boolean;
/**
 * Assert that a value or an array of value are positive integer.
 *
 * @param value The value being asserted on. May be a single number or an array
 *   of numbers.
 * @param name Name of the value, used to make the error message.
 */
declare function assertPositiveInteger(value: number | number[], name: string): void;
/**
 * Format a value into a display-friendly, human-readable fashion.
 *
 * - `null` is formatted as `'null'`
 * - Strings are formated with flanking pair of quotes.
 * - Arrays are formatted with flanking pair of square brackets.
 *
 * @param value The value to display.
 * @return Formatted string.
 */
declare function formatAsFriendlyString(value: any): string;
/**
 * Returns a function `f2` (decorator) which wraps the original function
 * `f`. `f2` guarantees that `f` can be called at most once
 * every `waitMs` ms. If `f2` is called more often, it will return
 * the last returned result of `f`.
 *
 * @param f The original function `f` to wrap.
 * @param waitMs The time between two consecutive calls to `f` in ms.
 */
declare function debounce<T>(f: (...args: Array<{}>) => T, waitMs: number, nowFunc?: Function): (...args: Array<{}>) => T;
/**
 * Returns the fusable activation given a layers identifier.
 *
 * @param activationName The layers identifier string.
 * @return The name of the fusable activation.
 */
declare function mapActivationToFusedKernel(activationName: string): fused.Activation;
declare type PossibleValues = Array<Array<boolean | string | number>>;
/**
 * Returns the cartesian product of sets of values.
 * This works the same as itertools.product in Python.
 *
 * Example:
 *
 * filters = [128, 256, 512]
 * paddings = ['same', 'valid']
 *
 * product = [ [128, 'same'], [128, 'valid'], [256, 'same'], [256, 'valid'],
 * [512, 'same'], [512, 'valid']]
 *
 * @param arrayOfValues List/array of values.
 * @return The cartesian product.
 */
declare function getCartesianProductOfValues(...arrayOfValues: PossibleValues): PossibleValues;
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/globals" />
/**
 * Enables production mode which disables correctness checks in favor of
 * performance.
 *
 * @doc {heading: 'Environment'}
 */
declare function enableProdMode(): void;
/**
 * Enables debug mode which will log information about all executed kernels:
 * the elapsed time of the kernel execution, as well as the rank, shape, and
 * size of the output tensor.
 *
 * Debug mode will significantly slow down your application as it will
 * download the result of every operation to the CPU. This should not be used in
 * production. Debug mode does not affect the timing information of the kernel
 * execution as we do not measure download time in the kernel execution time.
 *
 * See also: `tf.profile`, `tf.memory`.
 *
 * @doc {heading: 'Environment'}
 */
declare function enableDebugMode(): void;
/** Globally disables deprecation warnings */
declare function disableDeprecationWarnings(): void;
/** Warn users about deprecated functionality. */
declare function deprecationWarn(msg: string): void;
/**
 * Dispose all variables kept in backend engine.
 *
 * @doc {heading: 'Environment'}
 */
declare function disposeVariables(): void;
/**
 * It returns the global engine that keeps track of all tensors and backends.
 *
 * @doc {heading: 'Environment'}
 */
declare function engine(): Engine;
/**
 * Returns memory info at the current time in the program. The result is an
 * object with the following properties:
 *
 * - `numBytes`: Number of bytes allocated (undisposed) at this time.
 * - `numTensors`: Number of unique tensors allocated.
 * - `numDataBuffers`: Number of unique data buffers allocated
 *   (undisposed) at this time, which is ≤ the number of tensors
 *   (e.g. `a.reshape(newShape)` makes a new Tensor that shares the same
 *   data buffer with `a`).
 * - `unreliable`: True if the memory usage is unreliable. See `reasons` when
 *    `unreliable` is true.
 * - `reasons`: `string[]`, reasons why the memory is unreliable, present if
 *    `unreliable` is true.
 *
 * WebGL Properties:
 * - `numBytesInGPU`: Number of bytes allocated (undisposed) in the GPU only at
 *     this time.
 *
 * @doc {heading: 'Performance', subheading: 'Memory'}
 */
declare function memory(): MemoryInfo;
/**
 * Executes the provided function `f()` and returns a promise that resolves
 * with information about the function's memory use:
 * - `newBytes`: the number of new bytes allocated
 * - `newTensors`: the number of new tensors created
 * - `peakBytes`: the peak number of bytes allocated
 * - `kernels`: an array of objects for each kernel involved that reports
 * their input and output shapes, number of bytes used, and number of new
 * tensors created.
 * - `kernelNames`: an array of unique strings with just the names of the
 * kernels in the `kernels` array.
 *
 * ```js
 * const profile = await tf.profile(() => {
 *   const x = tf.tensor1d([1, 2, 3]);
 *   let x2 = x.square();
 *   x2.dispose();
 *   x2 = x.square();
 *   x2.dispose();
 *   return x;
 * });
 *
 * console.log(`newBytes: ${profile.newBytes}`);
 * console.log(`newTensors: ${profile.newTensors}`);
 * console.log(`byte usage over all kernels: ${profile.kernels.map(k =>
 * k.totalBytesSnapshot)}`);
 * ```
 *
 *
 * @doc {heading: 'Performance', subheading: 'Profile'}
 */
declare function profile(f: () => (TensorContainer | Promise<TensorContainer>)): Promise<ProfileInfo>;
/**
 * Executes the provided function `fn` and after it is executed, cleans up all
 * intermediate tensors allocated by `fn` except those returned by `fn`.
 * `fn` must not return a Promise (async functions not allowed). The returned
 * result can be a complex object.
 *
 * Using this method helps avoid memory leaks. In general, wrap calls to
 * operations in `tf.tidy` for automatic memory cleanup.
 *
 * NOTE: Variables do *not* get cleaned up when inside a tidy(). If you want to
 * dispose variables, please use `tf.disposeVariables` or call dispose()
 * directly on variables.
 *
 * ```js
 * // y = 2 ^ 2 + 1
 * const y = tf.tidy(() => {
 *   // a, b, and one will be cleaned up when the tidy ends.
 *   const one = tf.scalar(1);
 *   const a = tf.scalar(2);
 *   const b = a.square();
 *
 *   console.log('numTensors (in tidy): ' + tf.memory().numTensors);
 *
 *   // The value returned inside the tidy function will return
 *   // through the tidy, in this case to the variable y.
 *   return b.add(one);
 * });
 *
 * console.log('numTensors (outside tidy): ' + tf.memory().numTensors);
 * y.print();
 * ```
 *
 * @param nameOrFn The name of the closure, or the function to execute.
 *     If a name is provided, the 2nd argument should be the function.
 *     If debug mode is on, the timing and the memory usage of the function
 *     will be tracked and displayed on the console using the provided name.
 * @param fn The function to execute.
 *
 * @doc {heading: 'Performance', subheading: 'Memory'}
 */
declare function tidy<T extends TensorContainer>(nameOrFn: string | ScopeFn<T>, fn?: ScopeFn<T>): T;
/**
 * Disposes any `tf.Tensor`s found within the provided object.
 *
 * @param container an object that may be a `tf.Tensor` or may directly
 *     contain `tf.Tensor`s, such as a `Tensor[]` or `{key: Tensor, ...}`. If
 *     the object is not a `tf.Tensor` or does not contain `Tensors`, nothing
 *     happens. In general it is safe to pass any object here, except that
 *     `Promise`s are not supported.
 *
 * @doc {heading: 'Performance', subheading: 'Memory'}
 */
declare function dispose(container: TensorContainer): void;
/**
 * Keeps a `tf.Tensor` generated inside a `tf.tidy` from being disposed
 * automatically.
 *
 * ```js
 * let b;
 * const y = tf.tidy(() => {
 *   const one = tf.scalar(1);
 *   const a = tf.scalar(2);
 *
 *   // b will not be cleaned up by the tidy. a and one will be cleaned up
 *   // when the tidy ends.
 *   b = tf.keep(a.square());
 *
 *   console.log('numTensors (in tidy): ' + tf.memory().numTensors);
 *
 *   // The value returned inside the tidy function will return
 *   // through the tidy, in this case to the variable y.
 *   return b.add(one);
 * });
 *
 * console.log('numTensors (outside tidy): ' + tf.memory().numTensors);
 * console.log('y:');
 * y.print();
 * console.log('b:');
 * b.print();
 * ```
 *
 * @param result The tensor to keep from being disposed.
 *
 * @doc {heading: 'Performance', subheading: 'Memory'}
 */
declare function keep<T extends Tensor>(result: T): T;
/**
 * Executes `f()` and returns a promise that resolves with timing
 * information.
 *
 * The result is an object with the following properties:
 *
 * - `wallMs`: Wall execution time.
 * - `kernelMs`: Kernel execution time, ignoring data transfer. If using the
 * WebGL backend and the query timer extension is not available, this will
 * return an error object.
 * - On `WebGL` The following additional properties exist:
 *   - `uploadWaitMs`: CPU blocking time on texture uploads.
 *   - `downloadWaitMs`: CPU blocking time on texture downloads (readPixels).
 *
 * ```js
 * const x = tf.randomNormal([20, 20]);
 * const time = await tf.time(() => x.matMul(x));
 *
 * console.log(`kernelMs: ${time.kernelMs}, wallTimeMs: ${time.wallMs}`);
 * ```
 *
 * @param f The function to execute and time.
 *
 * @doc {heading: 'Performance', subheading: 'Timing'}
 */
declare function time(f: () => void): Promise<TimingInfo>;
/**
 * Sets the backend (cpu, webgl, wasm, etc) responsible for creating tensors and
 * executing operations on those tensors. Returns a promise that resolves
 * to a boolean if the backend initialization was successful.
 *
 * Note this disposes the current backend, if any, as well as any tensors
 * associated with it. A new backend is initialized, even if it is of the
 * same type as the previous one.
 *
 * @param backendName The name of the backend. Currently supports
 *     `'webgl'|'cpu'` in the browser, `'tensorflow'` under node.js
 *     (requires tfjs-node), and `'wasm'` (requires tfjs-backend-wasm).
 *
 * @doc {heading: 'Backends'}
 */
declare function setBackend(backendName: string): Promise<boolean>;
/**
 * Returns a promise that resolves when the currently selected backend (or the
 * highest priority one) has initialized. Await this promise when you are using
 * a backend that has async initialization.
 *
 * @doc {heading: 'Backends'}
 */
declare function ready(): Promise<void>;
/**
 * Returns the current backend name (cpu, webgl, etc). The backend is
 * responsible for creating tensors and executing operations on those tensors.
 *
 * @doc {heading: 'Backends'}
 */
declare function getBackend(): string;
/**
 * Removes a backend and the registered factory.
 *
 * @doc {heading: 'Backends'}
 */
declare function removeBackend(name: string): void;
/**
 * Finds the backend registered under the provided name. Returns null if the
 * name is not in the registry, or the registration hasn't finished yet.
 */
declare function findBackend(name: string): KernelBackend;
/**
 * Finds the backend factory registered under the provided name. Returns a
 * function that produces a new backend when called. Returns null if the name
 * is not in the registry.
 */
declare function findBackendFactory(name: string): () => KernelBackend | Promise<KernelBackend>;
/**
 * Registers a global backend. The registration should happen when importing
 * a module file (e.g. when importing `backend_webgl.ts`), and is used for
 * modular builds (e.g. custom tfjs bundle with only webgl support).
 *
 * @param factory The backend factory function. When called, it should
 * return a backend instance, or a promise of an instance.
 * @param priority The priority of the backend (higher = more important).
 *     In case multiple backends are registered, the priority is used to find
 *     the best backend. Defaults to 1.
 * @return False if there is already a registered backend under this name, true
 *     if not.
 *
 * @doc {heading: 'Backends'}
 */
declare function registerBackend(name: string, factory: () => KernelBackend | Promise<KernelBackend>, priority?: number): boolean;
/**
 * Gets the current backend. If no backends have been initialized, this will
 * attempt to initialize the best backend. Will throw an error if the highest
 * priority backend has async initialization, in which case you should call
 * 'await tf.ready()' before running other code.
 *
 * @doc {heading: 'Backends'}
 */
declare function backend(): KernelBackend;
/**
 * Sets the global platform.
 *
 * @param platformName The name of this platform.
 * @param platform A platform implementation.
 */
declare function setPlatform(platformName: string, platform: Platform): void;

/// <amd-module name="@tensorflow/tfjs-core/dist/globals_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/global_util" />
declare function getGlobalNamespace(): {
    _tfGlobals: Map<string, any>;
};
/**
 * Returns a globally accessible 'singleton' object.
 *
 * @param key the name of the object
 * @param init a function to initialize to initialize this object
 *             the first time it is fetched.
 */
declare function getGlobal<T>(key: string, init: () => T): T;

/// <amd-module name="@tensorflow/tfjs-core/dist/gradients" />
/**
 * Provided `f(x)`, returns another function `g(x, dy?)`, which gives the
 * gradient of `f(x)` with respect to `x`.
 *
 * If `dy` is provided, the gradient of `f(x).mul(dy).sum()` with respect to
 * `x` is computed instead. `f(x)` must take a single tensor `x` and return a
 * single tensor `y`. If `f()` takes multiple inputs, use `tf.grads` instead.
 *
 * ```js
 * // f(x) = x ^ 2
 * const f = x => x.square();
 * // f'(x) = 2x
 * const g = tf.grad(f);
 *
 * const x = tf.tensor1d([2, 3]);
 * g(x).print();
 * ```
 *
 * ```js
 * // f(x) = x ^ 3
 * const f = x => x.pow(tf.scalar(3, 'int32'));
 * // f'(x) = 3x ^ 2
 * const g = tf.grad(f);
 * // f''(x) = 6x
 * const gg = tf.grad(g);
 *
 * const x = tf.tensor1d([2, 3]);
 * gg(x).print();
 * ```
 *
 * @param f The function f(x), to compute gradient for.
 *
 * @doc {heading: 'Training', subheading: 'Gradients'}
 */
declare function grad(f: (x: Tensor) => Tensor): (x: TensorLike | Tensor, dy?: TensorLike | Tensor) => Tensor;
/**
 * Provided `f(x1, x2,...)`, returns another function `g([x1, x2,...], dy?)`,
 * which gives an array of gradients of `f()` with respect to each input
 * [`x1`,`x2`,...].
 *
 * If `dy` is passed when calling `g()`, the gradient of
 * `f(x1,...).mul(dy).sum()` with respect to each input is computed instead.
 * The provided `f` must take one or more tensors and return a single tensor
 * `y`. If `f()` takes a single input, we recommend using `tf.grad` instead.
 *
 * ```js
 * // f(a, b) = a * b
 * const f = (a, b) => a.mul(b);
 * // df / da = b, df / db = a
 * const g = tf.grads(f);
 *
 * const a = tf.tensor1d([2, 3]);
 * const b = tf.tensor1d([-2, -3]);
 * const [da, db] = g([a, b]);
 * console.log('da');
 * da.print();
 * console.log('db');
 * db.print();
 * ```
 *
 * @param f The function `f(x1, x2,...)` to compute gradients for.
 *
 * @doc {heading: 'Training', subheading: 'Gradients'}
 */
declare function grads(f: (...args: Tensor[]) => Tensor): (args: Array<Tensor | TensorLike>, dy?: Tensor | TensorLike) => Tensor[];
/**
 * Like `tf.grad`, but also returns the value of `f()`. Useful when `f()`
 * returns a metric you want to show.
 *
 * The result is a rich object with the following properties:
 * - grad: The gradient of `f(x)` w.r.t. `x` (result of `tf.grad`).
 * - value: The value returned by `f(x)`.
 *
 * ```js
 * // f(x) = x ^ 2
 * const f = x => x.square();
 * // f'(x) = 2x
 * const g = tf.valueAndGrad(f);
 *
 * const x = tf.tensor1d([2, 3]);
 * const {value, grad} = g(x);
 *
 * console.log('value');
 * value.print();
 * console.log('grad');
 * grad.print();
 * ```
 *
 * @doc {heading: 'Training', subheading: 'Gradients'}
 */
declare function valueAndGrad<I extends Tensor, O extends Tensor>(f: (x: I) => O): (x: I, dy?: O) => {
    value: O;
    grad: I;
};
/**
 * Like `tf.grads`, but returns also the value of `f()`. Useful when `f()`
 * returns a metric you want to show.
 *
 * The result is a rich object with the following properties:
 * - grads: The gradients of `f()` w.r.t. each input (result of `tf.grads`).
 * - value: The value returned by `f(x)`.
 *
 * ```js
 * // f(a, b) = a * b
 * const f = (a, b) => a.mul(b);
 * // df/da = b, df/db = a
 * const g = tf.valueAndGrads(f);
 *
 * const a = tf.tensor1d([2, 3]);
 * const b = tf.tensor1d([-2, -3]);
 * const {value, grads} = g([a, b]);
 *
 * const [da, db] = grads;
 *
 * console.log('value');
 * value.print();
 *
 * console.log('da');
 * da.print();
 * console.log('db');
 * db.print();
 * ```
 *
 * @doc {heading: 'Training', subheading: 'Gradients'}
 */
declare function valueAndGrads<O extends Tensor>(f: (...args: Tensor[]) => O): (args: Tensor[], dy?: O) => {
    grads: Tensor[];
    value: O;
};
/**
 * Computes and returns the gradient of f(x) with respect to the list of
 * trainable variables provided by `varList`. If no list is provided, it
 * defaults to all trainable variables.
 *
 * ```js
 * const a = tf.variable(tf.tensor1d([3, 4]));
 * const b = tf.variable(tf.tensor1d([5, 6]));
 * const x = tf.tensor1d([1, 2]);
 *
 * // f(a, b) = a * x ^ 2 + b * x
 * const f = () => a.mul(x.square()).add(b.mul(x)).sum();
 * // df/da = x ^ 2, df/db = x
 * const {value, grads} = tf.variableGrads(f);
 *
 * Object.keys(grads).forEach(varName => grads[varName].print());
 * ```
 *
 * @param f The function to execute. f() should return a scalar.
 * @param varList The list of variables to compute the gradients with respect
 *     to. Defaults to all trainable variables.
 * @returns An object with the following keys and values:
 *   - `value`: The value of the function `f`.
 *   - `grads`: A map from the names of the variables to the gradients.
 *     If the `varList` argument is provided explicitly and contains a subset of
 *     non-trainable variables, this map in the return value will contain keys
 *     that map the names of the non-trainable variables to `null`.
 *
 * @doc {heading: 'Training', subheading: 'Gradients'}
 */
declare function variableGrads(f: () => Scalar, varList?: Variable[]): {
    value: Scalar;
    grads: NamedTensorMap;
};
/**
 * Overrides the gradient computation of a function `f`.
 *
 * Takes a function
 * `f(...inputs, save) => {value: Tensor, gradFunc: (dy, saved) => Tensor[]}`
 * and returns another function `g(...inputs)` which takes the same inputs as
 * `f`. When called, `g` returns `f().value`. In backward mode, custom gradients
 * with respect to each input of `f` are computed using `f().gradFunc`.
 *
 * The `save` function passed to `f` should be used for saving tensors needed
 * in the gradient. And the `saved` passed to the `gradFunc` is a
 * `NamedTensorMap`, which contains those saved tensors.
 *
 * ```js
 * const customOp = tf.customGrad((x, save) => {
 *   // Save x to make sure it's available later for the gradient.
 *   save([x]);
 *   // Override gradient of our custom x ^ 2 op to be dy * abs(x);
 *   return {
 *     value: x.square(),
 *     // Note `saved.x` which points to the `x` we saved earlier.
 *     gradFunc: (dy, saved) => [dy.mul(saved[0].abs())]
 *   };
 * });
 *
 * const x = tf.tensor1d([-1, -2, 3]);
 * const dx = tf.grad(x => customOp(x));
 *
 * console.log(`f(x):`);
 * customOp(x).print();
 * console.log(`f'(x):`);
 * dx(x).print();
 * ```
 *
 * @param f The function to evaluate in forward mode, which should return
 *     `{value: Tensor, gradFunc: (dy, saved) => Tensor[]}`, where `gradFunc`
 *     returns the custom gradients of `f` with respect to its inputs.
 *
 * @doc {heading: 'Training', subheading: 'Gradients'}
 */
declare function customGrad<T extends Tensor>(f: CustomGradientFunc<T>): (...args: Tensor[]) => T;

/// <amd-module name="@tensorflow/tfjs-core/dist/gradients_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/linalg/gram_schmidt" />
/**
 * Gram-Schmidt orthogonalization.
 *
 * ```js
 * const x = tf.tensor2d([[1, 2], [3, 4]]);
 * let y = tf.linalg.gramSchmidt(x);
 * y.print();
 * console.log('Orthogonalized:');
 * y.dot(y.transpose()).print();  // should be nearly the identity matrix.
 * console.log('First row direction maintained:');
 * const data = await y.array();
 * console.log(data[0][1] / data[0][0]);  // should be nearly 2.
 * ```
 *
 * @param xs The vectors to be orthogonalized, in one of the two following
 *   formats:
 *   - An Array of `tf.Tensor1D`.
 *   - A `tf.Tensor2D`, i.e., a matrix, in which case the vectors are the rows
 *     of `xs`.
 *   In each case, all the vectors must have the same length and the length
 *   must be greater than or equal to the number of vectors.
 * @returns The orthogonalized and normalized vectors or matrix.
 *   Orthogonalization means that the vectors or the rows of the matrix
 *   are orthogonal (zero inner products). Normalization means that each
 *   vector or each row of the matrix has an L2 norm that equals `1`.
 *
 * @doc {heading:'Operations', subheading:'Linear Algebra', namespace:'linalg'}
 */
declare function gramSchmidt_(xs: Tensor1D[] | Tensor2D): Tensor1D[] | Tensor2D;
declare const gramSchmidt: typeof gramSchmidt_;
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/linalg/gram_schmidt_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/image/grayscale_to_rgb" />
/**
 * Converts images from grayscale to RGB format.
 *
 * @param image A grayscale tensor to convert. The `image`'s last dimension must
 *     be size 1 with at least a two-dimensional shape.
 *
 * @doc {heading: 'Operations', subheading: 'Images', namespace: 'image'}
 */
declare function grayscaleToRGB_<T extends Tensor2D | Tensor3D | Tensor4D | Tensor5D | Tensor6D>(image: T | TensorLike): T;
declare const grayscaleToRGB: typeof grayscaleToRGB_;
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/image/grayscale_to_rgb_test" />
{ };
/// <amd-module name="@tensorflow/tfjs-core/dist/ops/greater" />
/**
 * Returns the truth value of (a > b) element-wise. Supports broadcasting.
 *
 * ```js
 * const a = tf.tensor1d([1, 2, 3]);
 * const b = tf.tensor1d([2, 2, 2]);
 *
 * a.greater(b).print();
 * ```
 *
 * @param a The first input tensor.
 * @param b The second input tensor. Must have the same dtype as `a`.
 *
 * @doc {heading: 'Operations', subheading: 'Logical'}
 */
declare function greater_<T extends Tensor>(a: Tensor | TensorLike, b: Tensor | TensorLike): T;
declare const greater: typeof greater_;
{ };
/// <amd-module name="@tensorflow/tfjs-core/dist/gradients/GreaterEqual_grad" />
declare const greaterEqualGradConfig: GradConfig;
/// <amd-module name="@tensorflow/tfjs-core/dist/ops/greater_equal" />
/**
 * Returns the truth value of (a >= b) element-wise. Supports broadcasting.
 *
 * ```js
 * const a = tf.tensor1d([1, 2, 3]);
 * const b = tf.tensor1d([2, 2, 2]);
 *
 * a.greaterEqual(b).print();
 * ```
 *
 * @param a The first input tensor.
 * @param b The second input tensor. Must have the same dtype as `a`.
 *
 * @doc {heading: 'Operations', subheading: 'Logical'}
 */
declare function greaterEqual_<T extends Tensor>(a: Tensor | TensorLike, b: Tensor | TensorLike): T;
declare const greaterEqual: typeof greaterEqual_;
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/greater_equal_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/greater_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-data/dist/util/growing_ring_buffer" />
declare class GrowingRingBuffer<T> extends RingBuffer<T> {
    private static INITIAL_CAPACITY;
    /**
     * Constructs a `GrowingRingBuffer`.
     */
    constructor();
    isFull(): boolean;
    push(value: T): void;
    unshift(value: T): void;
    /**
     * Doubles the capacity of the buffer.
     */
    private expand;
}

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/signal/hamming_window" />
/**
 * Generate a hamming window.
 *
 * See: https://en.wikipedia.org/wiki/Window_function#Hann_and_Hamming_windows
 *
 * ```js
 * tf.signal.hammingWindow(10).print();
 * ```
 * @param The length of window
 *
 * @doc {heading: 'Operations', subheading: 'Signal', namespace: 'signal'}
 */
declare function hammingWindow_(windowLength: number): Tensor1D;
declare const hammingWindow: typeof hammingWindow_;
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/signal/hamming_window_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/signal/hann_window" />
/**
 * Generate a Hann window.
 *
 * See: https://en.wikipedia.org/wiki/Window_function#Hann_and_Hamming_windows
 *
 * ```js
 * tf.signal.hannWindow(10).print();
 * ```
 * @param The length of window
 *
 * @doc {heading: 'Operations', subheading: 'Signal', namespace: 'signal'}
 */
declare function hannWindow_(windowLength: number): Tensor1D;
declare const hannWindow: typeof hannWindow_;
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/signal/hann_window_test" />
{ };
/// <amd-module name="@tensorflow/tfjs-core/dist/hash_util" />
declare function hexToLong(hex: string): Long;
declare function fingerPrint64(s: Uint8Array, len?: number): Long;

/// <amd-module name="@tensorflow/tfjs-core/dist/hash_util_test" />
{ };
/// <amd-module name="@tensorflow/tfjs-core/dist/ops/losses/hinge_loss" />

/**
 * Computes the Hinge loss between two tensors.
 *
 * @param labels The ground truth output tensor, same dimensions as
 *    'predictions'.
 * @param predictions The predicted outputs.
 * @param weights Tensor whose rank is either 0, or the same rank as
 *    `labels`, and must be broadcastable to `labels` (i.e., all dimensions
 *    must be either `1`, or the same as the corresponding `losses`
 *    dimension).
 * @param reduction Type of reduction to apply to loss. Should be of type
 *    `Reduction`
 *
 * @doc {heading: 'Training', subheading: 'Losses', namespace: 'losses'}
 */
declare function hingeLoss_<T extends Tensor, O extends Tensor>(labels: T | TensorLike, predictions: T | TensorLike, weights?: Tensor | TensorLike, reduction?: Reduction): O;
declare const hingeLoss: typeof hingeLoss_;
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/losses/hinge_loss_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/io/http" />
declare class HTTPRequest implements IOHandler {
    protected readonly path: string;
    protected readonly requestInit: RequestInit;
    private readonly fetch;
    private readonly weightUrlConverter;
    readonly DEFAULT_METHOD = "POST";
    static readonly URL_SCHEME_REGEX: RegExp;
    private readonly weightPathPrefix;
    private readonly onProgress;
    constructor(path: string, loadOptions?: LoadOptions);
    save(modelArtifacts: ModelArtifacts): Promise<SaveResult>;
    /**
     * Load model artifacts via HTTP request(s).
     *
     * See the documentation to `tf.io.http` for details on the saved
     * artifacts.
     *
     * @returns The loaded model artifacts (if loading succeeds).
     */
    load(): Promise<ModelArtifacts>;
    private loadWeights;
}
/**
 * Extract the prefix and suffix of the url, where the prefix is the path before
 * the last file, and suffix is the search params after the last file.
 * ```
 * const url = 'http://tfhub.dev/model/1/tensorflowjs_model.pb?tfjs-format=file'
 * [prefix, suffix] = parseUrl(url)
 * // prefix = 'http://tfhub.dev/model/1/'
 * // suffix = '?tfjs-format=file'
 * ```
 * @param url the model url to be parsed.
 */
declare function parseUrl(url: string): [string, string];
declare function isHTTPScheme(url: string): boolean;
declare const httpRouter: IORouter;
/**
 * Creates an IOHandler subtype that sends model artifacts to HTTP server.
 *
 * An HTTP request of the `multipart/form-data` mime type will be sent to the
 * `path` URL. The form data includes artifacts that represent the topology
 * and/or weights of the model. In the case of Keras-style `tf.Model`, two
 * blobs (files) exist in form-data:
 *   - A JSON file consisting of `modelTopology` and `weightsManifest`.
 *   - A binary weights file consisting of the concatenated weight values.
 * These files are in the same format as the one generated by
 * [tfjs_converter](https://js.tensorflow.org/tutorials/import-keras.html).
 *
 * The following code snippet exemplifies the client-side code that uses this
 * function:
 *
 * ```js
 * const model = tf.sequential();
 * model.add(
 *     tf.layers.dense({units: 1, inputShape: [100], activation: 'sigmoid'}));
 *
 * const saveResult = await model.save(tf.io.http(
 *     'http://model-server:5000/upload', {requestInit: {method: 'PUT'}}));
 * console.log(saveResult);
 * ```
 *
 * If the default `POST` method is to be used, without any custom parameters
 * such as headers, you can simply pass an HTTP or HTTPS URL to `model.save`:
 *
 * ```js
 * const saveResult = await model.save('http://model-server:5000/upload');
 * ```
 *
 * The following GitHub Gist
 * https://gist.github.com/dsmilkov/1b6046fd6132d7408d5257b0976f7864
 * implements a server based on [flask](https://github.com/pallets/flask) that
 * can receive the request. Upon receiving the model artifacts via the requst,
 * this particular server reconstitutes instances of [Keras
 * Models](https://keras.io/models/model/) in memory.
 *
 *
 * @param path A URL path to the model.
 *   Can be an absolute HTTP path (e.g.,
 *   'http://localhost:8000/model-upload)') or a relative path (e.g.,
 *   './model-upload').
 * @param requestInit Request configurations to be used when sending
 *    HTTP request to server using `fetch`. It can contain fields such as
 *    `method`, `credentials`, `headers`, `mode`, etc. See
 *    https://developer.mozilla.org/en-US/docs/Web/API/Request/Request
 *    for more information. `requestInit` must not have a body, because the
 * body will be set by TensorFlow.js. File blobs representing the model
 * topology (filename: 'model.json') and the weights of the model (filename:
 * 'model.weights.bin') will be appended to the body. If `requestInit` has a
 * `body`, an Error will be thrown.
 * @param loadOptions Optional configuration for the loading. It includes the
 *   following fields:
 *   - weightPathPrefix Optional, this specifies the path prefix for weight
 *     files, by default this is calculated from the path param.
 *   - fetchFunc Optional, custom `fetch` function. E.g., in Node.js,
 *     the `fetch` from node-fetch can be used here.
 *   - onProgress Optional, progress callback function, fired periodically
 *     before the load is completed.
 * @returns An instance of `IOHandler`.
 *
 * @doc {
 *   heading: 'Models',
 *   subheading: 'Loading',
 *   namespace: 'io',
 *   ignoreCI: true
 * }
 */
declare function http(path: string, loadOptions?: LoadOptions): IOHandler;
/**
 * Deprecated. Use `tf.io.http`.
 * @param path
 * @param loadOptions
 */
declare function browserHTTPRequest(path: string, loadOptions?: LoadOptions): IOHandler;

/// <amd-module name="@tensorflow/tfjs-core/dist/io/http_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/losses/huber_loss" />
/**
 * Computes the Huber loss between two tensors.
 *
 * @param labels The ground truth output tensor, same dimensions as
 *    'predictions'.
 * @param predictions The predicted outputs.
 * @param weights Tensor whose rank is either 0, or the same rank as
 *    `labels`, and must be broadcastable to `labels` (i.e., all dimensions
 *    must be either `1`, or the same as the corresponding `losses`
 *    dimension).
 * @param delta Point where Huber loss changes from quadratic to linear.
 * @param reduction Type of reduction to apply to loss. Should be of type
 *    `Reduction`.
 *
 * @doc {heading: 'Training', subheading: 'Losses', namespace: 'losses'}
 */
declare function huberLoss_<T extends Tensor, O extends Tensor>(labels: T | TensorLike, predictions: T | TensorLike, weights?: Tensor | TensorLike, delta?: number, reduction?: Reduction): O;
declare const huberLoss: typeof huberLoss_;
{ };
/// <amd-module name="@tensorflow/tfjs-core/dist/ops/losses/huber_loss_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/gradients/Identity_grad" />
declare const identityGradConfig: GradConfig;

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/identity_pool_test" />
/**
 * Test utility for testing AvgPool, MaxPool, etc where kernel size is 1x1,
 * effectively making them act as the identity function except where strides
 * affect the output.
 */
declare function identityPoolTest(pool: typeof tf.avgPool): void;

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/spectral/ifft" />
/**
 * Inverse fast Fourier transform.
 *
 * Computes the inverse 1-dimensional discrete Fourier transform over the
 * inner-most dimension of input.
 *
 * ```js
 * const real = tf.tensor1d([1, 2, 3]);
 * const imag = tf.tensor1d([1, 2, 3]);
 * const x = tf.complex(real, imag);
 *
 * x.ifft().print();  // tf.spectral.ifft(x).print();
 * ```
 * @param input The complex input to compute an ifft over.
 *
 * @doc {heading: 'Operations', subheading: 'Spectral', namespace: 'spectral'}
 */
declare function ifft_(input: Tensor): Tensor;
declare const ifft: typeof ifft_;
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/ifft_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/imag" />
/**
 * Returns the imaginary part of a complex (or real) tensor.
 *
 * Given a tensor input, this operation returns a tensor of type float that is
 * the imaginary part of each element in input considered as a complex number.
 * If input is real, a tensor of all zeros is returned.
 *
 * ```js
 * const x = tf.complex([-2.25, 3.25], [4.75, 5.75]);
 * tf.imag(x).print();
 * ```
 *
 * @doc {heading: 'Tensors', subheading: 'Creation'}
 */
declare function imag_<T extends Tensor>(input: T | TensorLike): T;
declare const imag: typeof imag_;
{ };

/// <amd-module name="@tensorflow/tfjs-layers/dist/layers/preprocessing/image_preprocessing" />
declare interface RescalingArgs extends LayerArgs {
    scale: number;
    offset?: number;
}
/**
 * Preprocessing Rescaling Layer
 *
 * This rescales images by a scaling and offset factor
 */
declare class Rescaling extends Layer {
    /** @nocollapse */
    static className: string;
    private readonly scale;
    private readonly offset;
    constructor(args: RescalingArgs);
    getConfig(): serialization.ConfigDict;
    call(inputs: Tensor | Tensor[], kwargs: Kwargs): Tensor[] | Tensor;
}

/// <amd-module name="@tensorflow/tfjs-layers/dist/layers/preprocessing/image_resizing" />
declare const INTERPOLATION_KEYS: readonly ["bilinear", "nearest"];
declare type InterpolationType = typeof INTERPOLATION_KEYS[number];
declare interface ResizingArgs extends LayerArgs {
    height: number;
    width: number;
    interpolation?: InterpolationType;
    cropToAspectRatio?: boolean;
}
/**
 * Preprocessing Resizing Layer
 *
 * This resizes images by a scaling and offset factor
 */
declare class Resizing extends Layer {
    /** @nocollapse */
    static className: string;
    private readonly height;
    private readonly width;
    private readonly interpolation;
    private readonly cropToAspectRatio;
    constructor(args: ResizingArgs);
    computeOutputShape(inputShape: Shape | Shape[]): Shape | Shape[];
    getConfig(): serialization.ConfigDict;
    call(inputs: Tensor<Rank.R3> | Tensor<Rank.R4>, kwargs: Kwargs): Tensor[] | Tensor;
}
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/image_test_util" />
/**
 * Returns an image used in various image related tests as a 4d tensor.
 *
 * The image is 8x8 and looks like this:
 * https://drive.google.com/file/d/1Y0AsFZ2w9HsWgJfm8f2uDOGY7A4IHjcK/view?usp=sharing
 *
 */
declare function getTestImageAsTensor4d(): tf.Tensor4D;



/// <amd-module name="@tensorflow/tfjs-core/dist/io/indexed_db" />
/**
 * Delete the entire database for tensorflow.js, including the models store.
 */
declare function deleteDatabase(): Promise<void>;
/**
 * IOHandler subclass: Browser IndexedDB.
 *
 * See the doc string of `browserIndexedDB` for more details.
 */
declare class BrowserIndexedDB implements IOHandler {
    protected readonly indexedDB: IDBFactory;
    protected readonly modelPath: string;
    static readonly URL_SCHEME = "indexeddb://";
    constructor(modelPath: string);
    save(modelArtifacts: ModelArtifacts): Promise<SaveResult>;
    load(): Promise<ModelArtifacts>;
    /**
     * Perform database action to put model artifacts into or read model artifacts
     * from IndexedDB object store.
     *
     * Whether the action is put or get depends on whether `modelArtifacts` is
     * specified. If it is specified, the action will be put; otherwise the action
     * will be get.
     *
     * @param modelPath A unique string path for the model.
     * @param modelArtifacts If specified, it will be the model artifacts to be
     *   stored in IndexedDB.
     * @returns A `Promise` of `SaveResult`, if the action is put, or a `Promise`
     *   of `ModelArtifacts`, if the action is get.
     */
    private databaseAction;
}
declare const indexedDBRouter: IORouter;
/**
 * Creates a browser IndexedDB IOHandler for saving and loading models.
 *
 * ```js
 * const model = tf.sequential();
 * model.add(
 *     tf.layers.dense({units: 1, inputShape: [100], activation: 'sigmoid'}));
 *
 * const saveResult = await model.save('indexeddb://MyModel'));
 * console.log(saveResult);
 * ```
 *
 * @param modelPath A unique identifier for the model to be saved. Must be a
 *   non-empty string.
 * @returns An instance of `BrowserIndexedDB` (sublcass of `IOHandler`),
 *   which can be used with, e.g., `tf.Model.save`.
 */
declare function browserIndexedDB(modelPath: string): IOHandler;
declare class BrowserIndexedDBManager implements ModelStoreManager {
    private indexedDB;
    constructor();
    listModels(): Promise<{
        [path: string]: ModelArtifactsInfo;
    }>;
    removeModel(path: string): Promise<ModelArtifactsInfo>;
}

/// <amd-module name="@tensorflow/tfjs-core/dist/io/indexed_db_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-layers/dist/initializers" />
declare function checkFanMode(value?: string): void;
declare function checkDistribution(value?: string): void;
/**
 * Initializer base class.
 *
 * @doc {
 *   heading: 'Initializers', subheading: 'Classes', namespace: 'initializers'}
 */
declare abstract class Initializer extends serialization.Serializable {
    fromConfigUsesCustomObjects(): boolean;
    /**
     * Generate an initial value.
     * @param shape
     * @param dtype
     * @return The init value.
     */
    abstract apply(shape: Shape, dtype?: DataType): Tensor;
    getConfig(): serialization.ConfigDict;
}
declare class Zeros extends Initializer {
    /** @nocollapse */
    static className: string;
    apply(shape: Shape, dtype?: DataType): Tensor;
}
declare class Ones extends Initializer {
    /** @nocollapse */
    static className: string;
    apply(shape: Shape, dtype?: DataType): Tensor;
}
interface ConstantArgs {
    /** The value for each element in the variable. */
    value: number;
}
declare class Constant extends Initializer {
    /** @nocollapse */
    static className: string;
    private value;
    constructor(args: ConstantArgs);
    apply(shape: Shape, dtype?: DataType): Tensor;
    getConfig(): serialization.ConfigDict;
}
interface RandomUniformArgs {
    /** Lower bound of the range of random values to generate. */
    minval?: number;
    /** Upper bound of the range of random values to generate. */
    maxval?: number;
    /** Used to seed the random generator. */
    seed?: number;
}
declare class RandomUniform extends Initializer {
    /** @nocollapse */
    static className: string;
    readonly DEFAULT_MINVAL = -0.05;
    readonly DEFAULT_MAXVAL = 0.05;
    private minval;
    private maxval;
    private seed;
    constructor(args: RandomUniformArgs);
    apply(shape: Shape, dtype?: DataType): Tensor;
    getConfig(): serialization.ConfigDict;
}
interface RandomNormalArgs {
    /** Mean of the random values to generate. */
    mean?: number;
    /** Standard deviation of the random values to generate. */
    stddev?: number;
    /** Used to seed the random generator. */
    seed?: number;
}
declare class RandomNormal extends Initializer {
    /** @nocollapse */
    static className: string;
    readonly DEFAULT_MEAN = 0;
    readonly DEFAULT_STDDEV = 0.05;
    private mean;
    private stddev;
    private seed;
    constructor(args: RandomNormalArgs);
    apply(shape: Shape, dtype?: DataType): Tensor;
    getConfig(): serialization.ConfigDict;
}
interface TruncatedNormalArgs {
    /** Mean of the random values to generate. */
    mean?: number;
    /** Standard deviation of the random values to generate. */
    stddev?: number;
    /** Used to seed the random generator. */
    seed?: number;
}
declare class TruncatedNormal extends Initializer {
    /** @nocollapse */
    static className: string;
    readonly DEFAULT_MEAN = 0;
    readonly DEFAULT_STDDEV = 0.05;
    private mean;
    private stddev;
    private seed;
    constructor(args: TruncatedNormalArgs);
    apply(shape: Shape, dtype?: DataType): Tensor;
    getConfig(): serialization.ConfigDict;
}
interface IdentityArgs {
    /**
     * Multiplicative factor to apply to the identity matrix.
     */
    gain?: number;
}
declare class Identity extends Initializer {
    /** @nocollapse */
    static className: string;
    private gain;
    constructor(args: IdentityArgs);
    apply(shape: Shape, dtype?: DataType): Tensor;
    getConfig(): serialization.ConfigDict;
}
interface VarianceScalingArgs {
    /** Scaling factor (positive float). */
    scale?: number;
    /** Fanning mode for inputs and outputs. */
    mode?: FanMode;
    /** Probabilistic distribution of the values. */
    distribution?: Distribution;
    /** Random number generator seed. */
    seed?: number;
}
declare class VarianceScaling extends Initializer {
    /** @nocollapse */
    static className: string;
    private scale;
    private mode;
    private distribution;
    private seed;
    /**
     * Constructor of VarianceScaling.
     * @throws ValueError for invalid value in scale.
     */
    constructor(args: VarianceScalingArgs);
    apply(shape: Shape, dtype?: DataType): Tensor;
    getConfig(): serialization.ConfigDict;
}
interface SeedOnlyInitializerArgs {
    /** Random number generator seed. */
    seed?: number;
}
declare class GlorotUniform extends VarianceScaling {
    /** @nocollapse */
    static className: string;
    /**
     * Constructor of GlorotUniform
     * @param scale
     * @param mode
     * @param distribution
     * @param seed
     */
    constructor(args?: SeedOnlyInitializerArgs);
    getClassName(): string;
}
declare class GlorotNormal extends VarianceScaling {
    /** @nocollapse */
    static className: string;
    /**
     * Constructor of GlorotNormal.
     * @param scale
     * @param mode
     * @param distribution
     * @param seed
     */
    constructor(args?: SeedOnlyInitializerArgs);
    getClassName(): string;
}
declare class HeNormal extends VarianceScaling {
    /** @nocollapse */
    static className: string;
    constructor(args?: SeedOnlyInitializerArgs);
    getClassName(): string;
}
declare class HeUniform extends VarianceScaling {
    /** @nocollapse */
    static className: string;
    constructor(args?: SeedOnlyInitializerArgs);
    getClassName(): string;
}
declare class LeCunNormal extends VarianceScaling {
    /** @nocollapse */
    static className: string;
    constructor(args?: SeedOnlyInitializerArgs);
    getClassName(): string;
}
declare class LeCunUniform extends VarianceScaling {
    /** @nocollapse */
    static className: string;
    constructor(args?: SeedOnlyInitializerArgs);
    getClassName(): string;
}
interface OrthogonalArgs extends SeedOnlyInitializerArgs {
    /**
     * Multiplicative factor to apply to the orthogonal matrix. Defaults to 1.
     */
    gain?: number;
}
declare class Orthogonal extends Initializer {
    /** @nocollapse */
    static className: string;
    readonly DEFAULT_GAIN = 1;
    protected readonly gain: number;
    protected readonly seed: number;
    constructor(args?: OrthogonalArgs);
    apply(shape: Shape, dtype?: DataType): Tensor;
    getConfig(): serialization.ConfigDict;
}
/** @docinline */
declare type InitializerIdentifier = 'constant' | 'glorotNormal' | 'glorotUniform' | 'heNormal' | 'heUniform' | 'identity' | 'leCunNormal' | 'leCunUniform' | 'ones' | 'orthogonal' | 'randomNormal' | 'randomUniform' | 'truncatedNormal' | 'varianceScaling' | 'zeros' | string;
declare const INITIALIZER_IDENTIFIER_REGISTRY_SYMBOL_MAP: {
    [identifier in InitializerIdentifier]: string;
};
declare function serializeInitializer(initializer: Initializer): serialization.ConfigDictValue;
declare function getInitializer(identifier: InitializerIdentifier | Initializer | serialization.ConfigDict): Initializer;

/// <amd-module name="@tensorflow/tfjs-layers/dist/keras_format/initializer_config" />
/** @docinline */
declare type FanMode = 'fanIn' | 'fanOut' | 'fanAvg';
declare const VALID_FAN_MODE_VALUES: string[];
declare type FanModeSerialization = 'fan_in' | 'fan_out' | 'fan_avg';
/** @docinline */
declare type Distribution = 'normal' | 'uniform' | 'truncatedNormal';
declare const VALID_DISTRIBUTION_VALUES: string[];
declare type DistributionSerialization = 'normal' | 'uniform' | 'truncated_normal';
declare type ZerosSerialization = BaseSerialization<'Zeros', {}>;
declare type OnesSerialization = BaseSerialization<'Ones', {}>;
declare type ConstantConfig = {
    value: number;
};
declare type ConstantSerialization = BaseSerialization<'Constant', ConstantConfig>;
declare type RandomNormalConfig = {
    mean?: number;
    stddev?: number;
    seed?: number;
};
declare type RandomNormalSerialization = BaseSerialization<'RandomNormal', RandomNormalConfig>;
declare type RandomUniformConfig = {
    minval?: number;
    maxval?: number;
    seed?: number;
};
declare type RandomUniformSerialization = BaseSerialization<'RandomUniform', RandomUniformConfig>;
declare type TruncatedNormalConfig = {
    mean?: number;
    stddev?: number;
    seed?: number;
};
declare type TruncatedNormalSerialization = BaseSerialization<'TruncatedNormal', TruncatedNormalConfig>;
declare type VarianceScalingConfig = {
    scale?: number;
    mode?: FanModeSerialization;
    distribution?: DistributionSerialization;
    seed?: number;
};
declare type VarianceScalingSerialization = BaseSerialization<'VarianceScaling', VarianceScalingConfig>;
declare type OrthogonalConfig = {
    seed?: number;
    gain?: number;
};
declare type OrthogonalSerialization = BaseSerialization<'Orthogonal', OrthogonalConfig>;
declare type IdentityConfig = {
    gain?: number;
};
declare type IdentitySerialization = BaseSerialization<'Identity', IdentityConfig>;
declare type InitializerSerialization = ZerosSerialization | OnesSerialization | ConstantSerialization | RandomUniformSerialization | RandomNormalSerialization | TruncatedNormalSerialization | IdentitySerialization | VarianceScalingSerialization | OrthogonalSerialization;
declare type InitializerClassName = InitializerSerialization['class_name'];
/**
 * A string array of valid Initializer class names.
 *
 * This is guaranteed to match the `InitializerClassName` union type.
 */
declare const initializerClassNames: InitializerClassName[];

/// <amd-module name="@tensorflow/tfjs-layers/dist/keras_format/input_config" />
declare type InputLayerConfig = {
    name?: string;
    input_shape?: Shape;
    batch_size?: number;
    batch_input_shape?: Shape;
    dtype?: DataType;
    sparse?: boolean;
};
declare type InputLayerSerialization = BaseLayerSerialization<'InputLayer', InputLayerConfig>;
declare type InputLayerClassName = InputLayerSerialization['class_name'];
/**
 * A string array of valid InputLayer class names.
 *
 * This is guaranteed to match the `InputLayerClassName` union type.
 */
declare const inputLayerClassNames: InputLayerClassName[];

/// <amd-module name="@tensorflow/tfjs-layers/dist/engine/input_layer" />
/**
 * Constructor arguments for InputLayer.
 *
 * Note: You should provide only inputShape or batchInputShape (not both).
 * If only inputShape is provided, then the batchInputShape is determined by
 * the batchSize argument and the inputShape: [batchSize].concat(inputShape).
 */
declare interface InputLayerArgs {
    /** Input shape, not including the batch axis. */
    inputShape?: Shape;
    /** Optional input batch size (integer or null). */
    batchSize?: number;
    /** Batch input shape, including the batch axis. */
    batchInputShape?: Shape;
    /** Datatype of the input.  */
    dtype?: DataType;
    /**
     * Whether the placeholder created is meant to be sparse.
     */
    sparse?: boolean;
    /** Name of the layer. */
    name?: string;
}
declare class InputLayer extends Layer {
    /** @nocollapse */
    static readonly className = "InputLayer";
    sparse: boolean;
    constructor(args: InputLayerArgs);
    apply(inputs: Tensor | Tensor[] | SymbolicTensor | SymbolicTensor[], kwargs?: Kwargs): Tensor | Tensor[] | SymbolicTensor;
    dispose(): DisposeResult;
    getConfig(): serialization.ConfigDict;
}
/**
 * Config for the Input function.
 *
 * Note: You should provide only shape or batchShape (not both).
 * If only shape is provided, then the batchShape becomes
 * [null].concat(inputShape).
 */
interface InputConfig {
    /**
     * A shape, not including the batch size. For instance, `shape=[32]`
     * indicates that the expected input will be batches of 32-dimensional
     * vectors.
     */
    shape?: Shape;
    /**
     * A shape tuple (integer), including the batch size. For instance,
     * `batchShape=[10, 32]` indicates that the expected input will be batches of
     * 10 32-dimensional vectors. `batchShape=[null, 32]` indicates batches of an
     * arbitrary number of 32-dimensional vectors.
     */
    batchShape?: Shape;
    /**
     * An optional name string for the layer. Should be unique in a model (do not
     * reuse the same name twice). It will be autogenerated if it isn't provided.
     */
    name?: string;
    dtype?: DataType;
    /**
     * A boolean specifying whether the placeholder to be created is sparse.
     */
    sparse?: boolean;
}
declare function Input(config: InputConfig): SymbolicTensor;

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/in_top_k" />
/**
 * Returns whether the targets are in the top K predictions.
 *
 * ```js
 * const predictions = tf.tensor2d([[20, 10, 40, 30], [30, 50, -20, 10]]);
 * const targets = tf.tensor1d([2, 0]);
 * const precision = await tf.inTopKAsync(predictions, targets);
 * precision.print();
 * ```
 * @param predictions 2-D or higher `tf.Tensor` with last dimension being
 *     at least `k`.
 * @param targets 1-D or higher `tf.Tensor`.
 * @param k Optional Number of top elements to look at for computing precision,
 *     default to 1.
 *
 * @doc {heading: 'Operations', subheading: 'Evaluation'}
 */
declare function inTopKAsync_<T extends Tensor, U extends Tensor>(predictions: T | TensorLike, targets: U | TensorLike, k?: number): Promise<U>;
declare const inTopKAsync: typeof inTopKAsync_;
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/in_top_k_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/io/io" />

/// <amd-module name="@tensorflow/tfjs-core/dist/io/io_utils" />
/**
 * Encode a map from names to weight values as an ArrayBuffer, along with an
 * `Array` of `WeightsManifestEntry` as specification of the encoded weights.
 *
 * This function does not perform sharding.
 *
 * This function is the reverse of `decodeWeights`.
 *
 * @param tensors A map ("dict") from names to tensors.
 * @param group Group to which the weights belong (optional).
 * @returns A `Promise` of
 *   - A flat `ArrayBuffer` with all the binary values of the `Tensor`s
 *     concatenated.
 *   - An `Array` of `WeightManifestEntry`s, carrying information including
 *     tensor names, `dtype`s and shapes.
 * @throws Error: on unsupported tensor `dtype`.
 */
declare function encodeWeights(tensors: NamedTensorMap | NamedTensor[], group?: WeightGroup): Promise<{
    data: ArrayBuffer;
    specs: WeightsManifestEntry[];
}>;
/**
 * Decode flat ArrayBuffer as weights.
 *
 * This function does not handle sharding.
 *
 * This function is the reverse of `encodeWeights`.
 *
 * @param buffer A flat ArrayBuffer carrying the binary values of the tensors
 *   concatenated in the order specified in `specs`.
 * @param specs Specifications of the names, dtypes and shapes of the tensors
 *   whose value are encoded by `buffer`.
 * @return A map from tensor name to tensor value, with the names corresponding
 *   to names in `specs`.
 * @throws Error, if any of the tensors has unsupported dtype.
 */
declare function decodeWeights(buffer: ArrayBuffer, specs: WeightsManifestEntry[]): NamedTensorMap;
/**
 * Concatenate TypedArrays into an ArrayBuffer.
 */
declare function concatenateTypedArrays(xs: TypedArray[]): ArrayBuffer;
/**
 * Calculate the byte length of a JavaScript string.
 *
 * Note that a JavaScript string can contain wide characters, therefore the
 * length of the string is not necessarily equal to the byte length.
 *
 * @param str Input string.
 * @returns Byte length.
 */
declare function stringByteLength(str: string): number;
/**
 * Encode an ArrayBuffer as a base64 encoded string.
 *
 * @param buffer `ArrayBuffer` to be converted.
 * @returns A string that base64-encodes `buffer`.
 */
declare function arrayBufferToBase64String(buffer: ArrayBuffer): string;
/**
 * Decode a base64 string as an ArrayBuffer.
 *
 * @param str Base64 string.
 * @returns Decoded `ArrayBuffer`.
 */
declare function base64StringToArrayBuffer(str: string): ArrayBuffer;
/**
 * Concatenate a number of ArrayBuffers into one.
 *
 * @param buffers A number of array buffers to concatenate.
 * @returns Result of concatenating `buffers` in order.
 */
declare function concatenateArrayBuffers(buffers: ArrayBuffer[]): ArrayBuffer;
/**
 * Get the basename of a path.
 *
 * Behaves in a way analogous to Linux's basename command.
 *
 * @param path
 */
declare function basename(path: string): string;
/**
 * Create `ModelJSON` from `ModelArtifacts`.
 *
 * @param artifacts Model artifacts, describing the model and its weights.
 * @param manifest Weight manifest, describing where the weights of the
 *     `ModelArtifacts` are stored, and some metadata about them.
 * @returns Object representing the `model.json` file describing the model
 *     artifacts and weights
 */
declare function getModelJSONForModelArtifacts(artifacts: ModelArtifacts, manifest: WeightsManifestConfig): ModelJSON;
/**
 * Create `ModelArtifacts` from a JSON file and weights.
 *
 * @param modelJSON Object containing the parsed JSON of `model.json`
 * @param weightSpecs The list of WeightsManifestEntry for the model. Must be
 *     passed if the modelJSON has a weightsManifest.
 * @param weightData An ArrayBuffer of weight data for the model corresponding
 *     to the weights in weightSpecs. Must be passed if the modelJSON has a
 *     weightsManifest.
 * @returns A Promise of the `ModelArtifacts`, as described by the JSON file.
 */
declare function getModelArtifactsForJSONSync(modelJSON: ModelJSON, weightSpecs?: WeightsManifestEntry[], weightData?: ArrayBuffer): ModelArtifacts;
/**
 * Create `ModelArtifacts` from a JSON file.
 *
 * @param modelJSON Object containing the parsed JSON of `model.json`
 * @param loadWeights Function that takes the JSON file's weights manifest,
 *     reads weights from the listed path(s), and returns a Promise of the
 *     weight manifest entries along with the weights data.
 * @returns A Promise of the `ModelArtifacts`, as described by the JSON file.
 */
declare function getModelArtifactsForJSON(modelJSON: ModelJSON, loadWeights: (weightsManifest: WeightsManifestConfig) => Promise<[
    WeightsManifestEntry[],
    ArrayBuffer
]>): Promise<ModelArtifacts>;
/**
 * Populate ModelArtifactsInfo fields for a model with JSON topology.
 * @param modelArtifacts
 * @returns A ModelArtifactsInfo object.
 */
declare function getModelArtifactsInfoForJSON(modelArtifacts: ModelArtifacts): ModelArtifactsInfo;
/**
 * Concatenate the weights stored in a WeightsManifestConfig into a list of
 * WeightsManifestEntry
 *
 * @param weightsManifest The WeightsManifestConfig to extract weights from.
 * @returns A list of WeightsManifestEntry of the weights in the weightsManifest
 */
declare function getWeightSpecs(weightsManifest: WeightsManifestConfig): WeightsManifestEntry[];
/**
 * Retrieve a Float16 decoder which will decode a ByteArray of Float16 values
 * to a Float32Array.
 *
 * @returns Function (buffer: Uint16Array) => Float32Array which decodes
 *          the Uint16Array of Float16 bytes to a Float32Array.
 */
declare function getFloat16Decoder(): (buffer: Uint16Array) => Float32Array;

/// <amd-module name="@tensorflow/tfjs-core/dist/io/io_utils_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/spectral/irfft" />
/**
 * Inversed real value input fast Fourier transform.
 *
 * Computes the 1-dimensional inversed discrete Fourier transform over the
 * inner-most dimension of the real input.
 *
 * ```js
 * const real = tf.tensor1d([1, 2, 3]);
 * const imag = tf.tensor1d([0, 0, 0]);
 * const x = tf.complex(real, imag);
 *
 * x.irfft().print();
 * ```
 * @param input The real value input to compute an irfft over.
 *
 * @doc {heading: 'Operations', subheading: 'Spectral', namespace: 'spectral'}
 */
declare function irfft_(input: Tensor): Tensor;
declare const irfft: typeof irfft_;
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/spectral/irfft_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/gradients/IsFinite_grad" />
declare const isFiniteGradConfig: GradConfig;

/// <amd-module name="@tensorflow/tfjs-core/dist/gradients/IsInf_grad" />
declare const isInfGradConfig: GradConfig;

/// <amd-module name="@tensorflow/tfjs-core/dist/gradients/IsNan_grad" />
declare const isNanGradConfig: GradConfig;

/// <amd-module name="@tensorflow/tfjs-core/dist/public/chained_ops/is_finite" />
declare module '../../tensor' {
    interface Tensor<R extends Rank = Rank> {
        isFinite<T extends Tensor>(this: T): T;
    }
}

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/is_finite_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/is_inf" />
/**
 * Returns which elements of x are Infinity or -Infinity.
 *
 * ```js
 * const x = tf.tensor1d([NaN, Infinity, -Infinity, 0, 1]);
 *
 * x.isInf().print();  // or tf.isNaN(x)
 * ```
 * @param x The input Tensor.
 *
 * @doc {heading: 'Operations', subheading: 'Basic math'}
 */
declare function isInf_<T extends Tensor>(x: T | TensorLike): T;
declare const isInf: typeof isInf_;
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/is_inf_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/public/chained_ops/is_nan" />
declare module '../../tensor' {
    interface Tensor<R extends Rank = Rank> {
        isNaN<T extends Tensor>(this: T): T;
    }
}

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/is_nan_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/jasmine_util" />
declare type Constraints = {
    flags?: Flags;
    predicate?: (testEnv: TestEnv) => boolean;
};
declare const NODE_ENVS: Constraints;
declare const CHROME_ENVS: Constraints;
declare const BROWSER_ENVS: Constraints;
declare const SYNC_BACKEND_ENVS: Constraints;
declare const HAS_WORKER: {
    predicate: () => boolean;
};
declare const HAS_NODE_WORKER: {
    predicate: () => boolean;
};
declare const ALL_ENVS: Constraints;
declare function envSatisfiesConstraints(env: Environment, testEnv: TestEnv, constraints: Constraints): boolean;
interface TestFilter {
    include?: string;
    startsWith?: string;
    excludes?: string[];
}
/**
 * Add test filtering logic to Jasmine's specFilter hook.
 *
 * @param testFilters Used for include a test suite, with the ability
 *     to selectively exclude some of the tests.
 *     Either `include` or `startsWith` must exist for a `TestFilter`.
 *     Tests that have the substrings specified by the include or startsWith
 *     will be included in the test run, unless one of the substrings specified
 *     by `excludes` appears in the name.
 * @param customInclude Function to programatically include a test.
 *     If this function returns true, a test will immediately run. Otherwise,
 *     `testFilters` is used for fine-grained filtering.
 *
 * If a test is not handled by `testFilters` or `customInclude`, the test will
 * be excluded in the test run.
 */
declare function setupTestFilters(testFilters: TestFilter[], customInclude: (name: string) => boolean): void;
declare function parseTestEnvFromKarmaFlags(args: string[], registeredTestEnvs: TestEnv[]): TestEnv;
declare function describeWithFlags(name: string, constraints: Constraints, tests: (env: TestEnv) => void): void;
interface TestEnv {
    name: string;
    backendName: string;
    flags?: Flags;
    isDataSync?: boolean;
}
declare const TEST_ENVS: TestEnv[];
declare function setTestEnvs(testEnvs: TestEnv[]): void;
declare function registerTestEnv(testEnv: TestEnv): void;
declare class TestKernelBackend extends KernelBackend {
    dispose(): void;
}
/**
 * Wraps a Jasmine spec's test function so it is run exclusively to others that
 * use runWithLock.
 *
 * @param spec The function that runs the spec. Must return a promise or call
 *     `done()`.
 *
 */
declare function runWithLock(spec: (done?: DoneFn) => Promise<void> | void): () => Promise<void>;

/// <amd-module name="@tensorflow/tfjs-core/dist/jasmine_util_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-layers/dist/keras_format/keras_class_names" />
/**
 * A type representing all possible Serializations of Keras objects, including
 * Layers, Constraints, Optimizers, etc.
 */
declare type KerasSerialization = LayerSerialization | ConstraintSerialization | InitializerSerialization | RegularizerSerialization | OptimizerSerialization;
/**
 * A type representing all valid values of `class_name` in a Keras JSON file
 * (regardless of context, which will naturally further restrict the valid
 * values).
 */
declare type KerasClassName = KerasSerialization['class_name'];
declare const kerasClassNames: KerasClassName[];

/// <amd-module name="@tensorflow/tfjs-core/dist/backends/kernel_impls" />
/// <amd-module name="@tensorflow/tfjs-core/dist/kernel_names" />

declare const Abs = "Abs";
declare type AbsInputs = UnaryInputs;
declare const Acos = "Acos";
declare type AcosInputs = UnaryInputs;
declare const Acosh = "Acosh";
declare type AcoshInputs = UnaryInputs;
declare const Add = "Add";
declare type AddInputs = BinaryInputs;
declare const AddN = "AddN";
declare type AddNInputs = TensorInfo[];
declare const All = "All";
declare type AllInputs = Pick<NamedTensorInfoMap, 'x'>;
interface AllAttrs {
    axis: number | number[];
    keepDims: boolean;
}
declare const Any = "Any";
declare type AnyInputs = Pick<NamedTensorInfoMap, 'x'>;
interface AnyAttrs {
    axis: number | number[];
    keepDims: boolean;
}
declare const ArgMax = "ArgMax";
declare type ArgMaxInputs = Pick<NamedTensorInfoMap, 'x'>;
interface ArgMaxAttrs {
    axis: number;
}
declare const ArgMin = "ArgMin";
declare type ArgMinInputs = Pick<NamedTensorInfoMap, 'x'>;
interface ArgMinAttrs {
    axis: number;
}
declare const Asin = "Asin";
declare type AsinInputs = UnaryInputs;
declare const Asinh = "Asinh";
declare type AsinhInputs = UnaryInputs;
declare const Atan = "Atan";
declare type AtanInputs = UnaryInputs;
declare const Atanh = "Atanh";
declare type AtanhInputs = UnaryInputs;
declare const Atan2 = "Atan2";
declare type Atan2Inputs = BinaryInputs;
declare const AvgPool = "AvgPool";
declare type AvgPoolInputs = Pick<NamedTensorInfoMap, 'x'>;
interface AvgPoolAttrs {
    filterSize: [number, number] | number;
    strides: [number, number] | number;
    pad: 'valid' | 'same' | number | ExplicitPadding;
    dimRoundingMode?: 'floor' | 'round' | 'ceil';
}
declare const AvgPoolGrad = "AvgPoolGrad";
declare type AvgPoolGradInputs = Pick<NamedTensorInfoMap, 'dy' | 'input'>;
interface AvgPoolGradAttrs {
    filterSize: [number, number] | number;
    strides: [number, number] | number;
    pad: 'valid' | 'same' | number | ExplicitPadding;
}
declare const AvgPool3D = "AvgPool3D";
declare type AvgPool3DInputs = Pick<NamedTensorInfoMap, 'x'>;
interface AvgPool3DAttrs {
    filterSize: [number, number, number] | number;
    strides: [number, number, number] | number;
    pad: 'valid' | 'same' | number;
    dimRoundingMode?: 'floor' | 'round' | 'ceil';
    dataFormat: 'NDHWC' | 'NCDHW';
}
declare const AvgPool3DGrad = "AvgPool3DGrad";
declare type AvgPool3DGradInputs = Pick<NamedTensorInfoMap, 'dy' | 'input'>;
interface AvgPool3DGradAttrs {
    filterSize: [number, number, number] | number;
    strides: [number, number, number] | number;
    pad: 'valid' | 'same' | number;
    dimRoundingMode?: 'floor' | 'round' | 'ceil';
}
declare const BatchMatMul = "BatchMatMul";
declare type BatchMatMulInputs = Pick<NamedTensorInfoMap, 'a' | 'b'>;
interface BatchMatMulAttrs {
    transposeA: boolean;
    transposeB: boolean;
}
declare const BatchToSpaceND = "BatchToSpaceND";
declare type BatchToSpaceNDInputs = Pick<NamedTensorInfoMap, 'x'>;
interface BatchToSpaceNDAttrs {
    blockShape: number[];
    crops: number[][];
}
declare type BinaryInputs = Pick<NamedTensorInfoMap, 'a' | 'b'>;
declare const Bincount = "Bincount";
declare type BincountInputs = Pick<NamedTensorInfoMap, 'x' | 'weights'>;
interface BincountAttrs {
    size: number;
}
declare const BroadcastTo = "BroadcastTo";
declare type BroadcastToInputs = Pick<NamedTensorInfoMap, 'x'>;
interface BroadCastToAttrs {
    shape: number[];
    inputShape: number[];
}
declare const BroadcastArgs = "BroadcastArgs";
declare type BroadcastArgsInputs = Pick<NamedTensorInfoMap, 's0' | 's1'>;
declare const Cast = "Cast";
declare type CastInputs = UnaryInputs;
interface CastAttrs {
    dtype: DataType;
}
declare const Ceil = "Ceil";
declare type CeilInputs = UnaryInputs;
declare const ClipByValue = "ClipByValue";
declare type ClipByValueInputs = UnaryInputs;
interface ClipByValueAttrs {
    clipValueMin: number;
    clipValueMax: number;
}
declare const Complex = "Complex";
declare type ComplexInputs = Pick<NamedTensorInfoMap, 'real' | 'imag'>;
declare const ComplexAbs = "ComplexAbs";
declare type ComplexAbsInputs = UnaryInputs;
declare const Concat = "Concat";
declare type ConcatInputs = TensorInfo[];
interface ConcatAttrs {
    axis: number;
}
declare const Conv2D = "Conv2D";
declare type Conv2DInputs = Pick<NamedTensorInfoMap, 'x' | 'filter'>;
interface Conv2DAttrs {
    strides: [number, number] | number;
    pad: 'valid' | 'same' | number | ExplicitPadding;
    dataFormat: 'NHWC' | 'NCHW';
    dilations: [number, number] | number;
    dimRoundingMode?: 'floor' | 'round' | 'ceil';
}
declare const Conv2DBackpropFilter = "Conv2DBackpropFilter";
declare type Conv2DBackpropFilterInputs = Pick<NamedTensorInfoMap, 'x' | 'dy'>;
interface Conv2DBackpropFilterAttrs {
    strides: [number, number] | number;
    pad: 'valid' | 'same' | number | ExplicitPadding;
    dataFormat: 'NHWC' | 'NCHW';
    dimRoundingMode?: 'floor' | 'round' | 'ceil';
    filterShape: [number, number, number, number];
}
declare const Conv2DBackpropInput = "Conv2DBackpropInput";
declare type Conv2DBackpropInputInputs = Pick<NamedTensorInfoMap, 'dy' | 'filter'>;
interface Conv2DBackpropInputAttrs {
    strides: [number, number] | number;
    pad: 'valid' | 'same' | number | ExplicitPadding;
    dataFormat: 'NHWC' | 'NCHW';
    dimRoundingMode?: 'floor' | 'round' | 'ceil';
    inputShape: [number, number, number, number];
}
declare const Conv3D = "Conv3D";
declare type Conv3DInputs = Pick<NamedTensorInfoMap, 'x' | 'filter'>;
interface Conv3DAttrs {
    strides: [number, number, number] | number;
    pad: 'valid' | 'same';
    dataFormat: 'NDHWC' | 'NCDHW';
    dilations: [number, number, number] | number;
}
declare const Conv3DBackpropFilterV2 = "Conv3DBackpropFilterV2";
declare type Conv3DBackpropFilterV2Inputs = Pick<NamedTensorInfoMap, 'x' | 'dy'>;
interface Conv3DBackpropFilterV2Attrs {
    strides: [number, number, number] | number;
    pad: 'valid' | 'same';
    filterShape: [number, number, number, number, number];
}
declare const Conv3DBackpropInputV2 = "Conv3DBackpropInputV2";
declare type Conv3DBackpropInputV2Inputs = Pick<NamedTensorInfoMap, 'dy' | 'filter'>;
interface Conv3DBackpropInputV2Attrs {
    strides: [number, number, number] | number;
    pad: 'valid' | 'same';
    inputShape: [number, number, number, number, number];
}
declare const Cos = "Cos";
declare type CosInputs = UnaryInputs;
declare const Cosh = "Cosh";
declare type CoshInputs = UnaryInputs;
declare const Cumprod = "Cumprod";
declare type CumprodInputs = Pick<NamedTensorInfoMap, 'x'>;
interface CumprodAttrs {
    axis: number;
    exclusive: boolean;
    reverse: boolean;
}
declare const Cumsum = "Cumsum";
declare type CumsumInputs = Pick<NamedTensorInfoMap, 'x'>;
interface CumsumAttrs {
    axis: number;
    exclusive: boolean;
    reverse: boolean;
}
declare const CropAndResize = "CropAndResize";
declare type CropAndResizeInputs = Pick<NamedTensorInfoMap, 'image' | 'boxes' | 'boxInd'>;
interface CropAndResizeAttrs {
    cropSize: [number, number];
    method: 'bilinear' | 'nearest';
    extrapolationValue: number;
}
declare const DenseBincount = "DenseBincount";
declare type DenseBincountInputs = Pick<NamedTensorInfoMap, 'x' | 'weights'>;
interface DenseBincountAttrs {
    size: number;
    binaryOutput?: boolean;
}
declare const DepthToSpace = "DepthToSpace";
declare type DepthToSpaceInputs = Pick<NamedTensorInfoMap, 'x'>;
interface DepthToSpaceAttrs {
    blockSize: number;
    dataFormat: 'NHWC' | 'NCHW';
}
declare const DepthwiseConv2dNative = "DepthwiseConv2dNative";
declare type DepthwiseConv2dNativeInputs = Pick<NamedTensorInfoMap, 'x' | 'filter'>;
interface DepthwiseConv2dNativeAttrs {
    strides: [number, number] | number;
    pad: 'valid' | 'same' | number | ExplicitPadding;
    dataFormat: 'NHWC' | 'NCHW';
    dilations: [number, number] | number;
    dimRoundingMode?: 'floor' | 'round' | 'ceil';
}
declare const DepthwiseConv2dNativeBackpropFilter = "DepthwiseConv2dNativeBackpropFilter";
declare type DepthwiseConv2dNativeBackpropFilterInputs = Pick<NamedTensorInfoMap, 'x' | 'dy'>;
interface DepthwiseConv2dNativeBackpropFilterAttrs {
    strides: [number, number] | number;
    dilations: [number, number] | number;
    pad: 'valid' | 'same' | number | ExplicitPadding;
    dimRoundingMode?: 'floor' | 'round' | 'ceil';
    filterShape: [number, number, number, number];
}
declare const DepthwiseConv2dNativeBackpropInput = "DepthwiseConv2dNativeBackpropInput";
declare type DepthwiseConv2dNativeBackpropInputInputs = Pick<NamedTensorInfoMap, 'dy' | 'filter'>;
interface DepthwiseConv2dNativeBackpropInputAttrs {
    strides: [number, number] | number;
    dilations: [number, number] | number;
    pad: 'valid' | 'same' | number | ExplicitPadding;
    dimRoundingMode?: 'floor' | 'round' | 'ceil';
    inputShape: [number, number, number, number];
}
declare const Diag = "Diag";
declare type DiagInputs = Pick<NamedTensorInfoMap, 'x'>;
declare const Dilation2D = "Dilation2D";
declare type Dilation2DInputs = Pick<NamedTensorInfoMap, 'x' | 'filter'>;
interface Dilation2DAttrs {
    strides: [number, number] | number;
    pad: 'valid' | 'same' | number;
    dilations: [number, number] | number;
}
declare const Dilation2DBackpropInput = "Dilation2DBackpropInput";
declare type Dilation2DBackpropInputInputs = Pick<NamedTensorInfoMap, 'x' | 'filter' | 'dy'>;
declare const Dilation2DBackpropFilter = "Dilation2DBackpropFilter";
declare type Dilation2DBackpropFilterInputs = Pick<NamedTensorInfoMap, 'x' | 'filter' | 'dy'>;
declare const RealDiv = "RealDiv";
declare type RealDivInputs = BinaryInputs;
declare const Einsum = "Einsum";
declare type EinsumInputs = TensorInfo[];
interface EinsumAttrs {
    equation: string;
}
declare const Elu = "Elu";
declare type EluInputs = Pick<NamedTensorInfoMap, 'x'>;
declare const EluGrad = "EluGrad";
declare type EluGradInputs = Pick<NamedTensorInfoMap, 'dy' | 'y'>;
declare const Erf = "Erf";
declare type ErfInputs = UnaryInputs;
declare const Equal = "Equal";
declare type EqualInputs = BinaryInputs;
declare const Exp = "Exp";
declare type ExpInputs = UnaryInputs;
declare const ExpandDims = "ExpandDims";
declare type ExpandDimsInputs = Pick<NamedTensorInfoMap, 'input'>;
interface ExpandDimsAttrs {
    dim: number;
}
declare const Expm1 = "Expm1";
declare type Expm1Inputs = UnaryInputs;
declare const FFT = "FFT";
declare type FFTInputs = Pick<NamedTensorInfoMap, 'input'>;
declare const Fill = "Fill";
interface FillAttrs {
    shape: number[];
    value: number | string;
    dtype: DataType;
}
declare const FlipLeftRight = "FlipLeftRight";
declare type FlipLeftRightInputs = Pick<NamedTensorInfoMap, 'image'>;
declare const Floor = "Floor";
declare type FloorInputs = UnaryInputs;
declare const FloorDiv = "FloorDiv";
declare type FloorDivInputs = BinaryInputs;
declare const FusedBatchNorm = "FusedBatchNorm";
declare type FusedBatchNormInputs = Pick<NamedTensorInfoMap, 'x' | 'scale' | 'offset' | 'mean' | 'variance'>;
interface FusedBatchNormAttrs {
    varianceEpsilon: number;
}
declare const GatherV2 = "GatherV2";
declare type GatherV2Inputs = Pick<NamedTensorInfoMap, 'x' | 'indices'>;
interface GatherV2Attrs {
    axis: number;
    batchDims: number;
}
declare const GatherNd = "GatherNd";
declare type GatherNdInputs = Pick<NamedTensorInfoMap, 'params' | 'indices'>;
declare const Greater = "Greater";
declare type GreaterInputs = BinaryInputs;
declare const GreaterEqual = "GreaterEqual";
declare type GreaterEqualInputs = BinaryInputs;
declare const Identity = "Identity";
declare type IdentityInputs = Pick<NamedTensorInfoMap, 'x'>;
declare const IFFT = "IFFT";
declare type IFFTInputs = Pick<NamedTensorInfoMap, 'input'>;
declare const Imag = "Imag";
declare type ImagInputs = Pick<NamedTensorInfoMap, 'input'>;
declare const IsFinite = "IsFinite";
declare type IsFiniteInputs = UnaryInputs;
declare const IsInf = "IsInf";
declare type IsInfInputs = UnaryInputs;
declare const IsNan = "IsNan";
declare type IsNanInputs = UnaryInputs;
declare const LeakyRelu = "LeakyRelu";
declare type LeakyReluInputs = Pick<NamedTensorInfoMap, 'x'>;
interface LeakyReluAttrs {
    alpha: number;
}
declare const Less = "Less";
declare type LessInputs = BinaryInputs;
declare const LessEqual = "LessEqual";
declare type LessEqualInputs = BinaryInputs;
declare const LinSpace = "LinSpace";
interface LinSpaceAttrs {
    start: number;
    stop: number;
    num: number;
}
declare const Log = "Log";
declare type LogInputs = UnaryInputs;
declare const Log1p = "Log1p";
declare type Log1pInputs = UnaryInputs;
declare const LogicalAnd = "LogicalAnd";
declare type LogicalAndInputs = BinaryInputs;
declare const LogicalNot = "LogicalNot";
declare type LogicalNotInputs = Pick<NamedTensorInfoMap, 'x'>;
declare const LogicalOr = "LogicalOr";
declare type LogicalOrInputs = BinaryInputs;
declare const LogicalXor = "LogicalXor";
declare type LogicalXorInputs = BinaryInputs;
declare const LogSoftmax = "LogSoftmax";
declare type LogSoftmaxInputs = Pick<NamedTensorInfoMap, 'logits'>;
interface LogSoftmaxAttrs {
    axis: number;
}
declare const LowerBound = "LowerBound";
declare type LowerBoundInputs = Pick<NamedTensorInfoMap, 'sortedSequence' | 'values'>;
declare const LRN = "LRN";
declare type LRNInputs = Pick<NamedTensorInfoMap, 'x'>;
interface LRNAttrs {
    depthRadius: number;
    bias: number;
    alpha: number;
    beta: number;
}
declare const LRNGrad = "LRNGrad";
declare type LRNGradInputs = Pick<NamedTensorInfoMap, 'x' | 'y' | 'dy'>;
interface LRNGradAttrs {
    depthRadius: number;
    bias: number;
    alpha: number;
    beta: number;
}
declare const Max = "Max";
declare type MaxInputs = Pick<NamedTensorInfoMap, 'x'>;
interface MaxAttrs {
    reductionIndices: number | number[];
    keepDims: boolean;
}
declare const Maximum = "Maximum";
declare type MaximumInputs = BinaryInputs;
declare const MaxPool = "MaxPool";
declare type MaxPoolInputs = Pick<NamedTensorInfoMap, 'x'>;
interface MaxPoolAttrs {
    filterSize: [number, number] | number;
    strides: [number, number] | number;
    pad: 'valid' | 'same' | number | ExplicitPadding;
    dimRoundingMode?: 'floor' | 'round' | 'ceil';
}
declare const MaxPoolGrad = "MaxPoolGrad";
declare type MaxPoolGradInputs = Pick<NamedTensorInfoMap, 'dy' | 'input' | 'output'>;
interface MaxPoolGradAttrs {
    filterSize: [number, number] | number;
    strides: [number, number] | number;
    pad: 'valid' | 'same' | number | ExplicitPadding;
    dimRoundingMode?: 'floor' | 'round' | 'ceil';
}
declare const MaxPool3D = "MaxPool3D";
declare type MaxPool3DInputs = Pick<NamedTensorInfoMap, 'x'>;
interface MaxPool3DAttrs {
    filterSize: [number, number, number] | number;
    strides: [number, number, number] | number;
    pad: 'valid' | 'same' | number;
    dataFormat: 'NDHWC' | 'NCDHW';
    dimRoundingMode?: 'floor' | 'round' | 'ceil';
}
declare const MaxPool3DGrad = "MaxPool3DGrad";
declare type MaxPool3DGradInputs = Pick<NamedTensorInfoMap, 'dy' | 'input' | 'output'>;
interface MaxPool3DGradAttrs {
    filterSize: [number, number, number] | number;
    strides: [number, number, number] | number;
    pad: 'valid' | 'same' | number;
    dimRoundingMode?: 'floor' | 'round' | 'ceil';
}
declare const MaxPoolWithArgmax = "MaxPoolWithArgmax";
declare type MaxPoolWithArgmaxInputs = Pick<NamedTensorInfoMap, 'x'>;
interface MaxPoolWithArgmaxAttrs {
    filterSize: [number, number] | number;
    strides: [number, number] | number;
    pad: 'valid' | 'same' | number;
    includeBatchInIndex: boolean;
}
declare const Mean = "Mean";
declare type MeanInputs = Pick<NamedTensorInfoMap, 'x'>;
interface MeanAttrs {
    axis: number | number[];
    keepDims: boolean;
}
declare const Min = "Min";
declare type MinInputs = Pick<NamedTensorInfoMap, 'x'>;
interface MinAttrs {
    axis: number | number[];
    keepDims: boolean;
}
declare const Minimum = "Minimum";
declare type MinimumInputs = BinaryInputs;
declare const MirrorPad = "MirrorPad";
declare type MirrorPadInputs = Pick<NamedTensorInfoMap, 'x'>;
interface MirrorPadAttrs {
    paddings: Array<[number, number]>;
    mode: 'reflect' | 'symmetric';
}
declare const Mod = "Mod";
declare type ModInputs = BinaryInputs;
declare const Multinomial = "Multinomial";
declare type MultinomialInputs = Pick<NamedTensorInfoMap, 'logits'>;
interface MultinomialAttrs {
    numSamples: number;
    seed: number;
    normalized: boolean;
}
declare const Multiply = "Multiply";
declare type MultiplyInputs = BinaryInputs;
declare const Neg = "Neg";
declare type NegInputs = UnaryInputs;
declare const NotEqual = "NotEqual";
declare type NotEqualInputs = BinaryInputs;
declare const NonMaxSuppressionV3 = "NonMaxSuppressionV3";
declare type NonMaxSuppressionV3Inputs = Pick<NamedTensorInfoMap, 'boxes' | 'scores'>;
interface NonMaxSuppressionV3Attrs {
    maxOutputSize: number;
    iouThreshold: number;
    scoreThreshold: number;
}
declare const NonMaxSuppressionV4 = "NonMaxSuppressionV4";
declare type NonMaxSuppressionV4Inputs = Pick<NamedTensorInfoMap, 'boxes' | 'scores'>;
interface NonMaxSuppressionV4Attrs {
    maxOutputSize: number;
    iouThreshold: number;
    scoreThreshold: number;
    padToMaxOutputSize: boolean;
}
declare const NonMaxSuppressionV5 = "NonMaxSuppressionV5";
declare type NonMaxSuppressionV5Inputs = Pick<NamedTensorInfoMap, 'boxes' | 'scores'>;
interface NonMaxSuppressionV5Attrs {
    maxOutputSize: number;
    iouThreshold: number;
    scoreThreshold: number;
    softNmsSigma: number;
}
declare const OnesLike = "OnesLike";
declare type OnesLikeInputs = UnaryInputs;
declare const OneHot = "OneHot";
declare type OneHotInputs = Pick<NamedTensorInfoMap, 'indices'>;
interface OneHotAttrs {
    depth: number;
    onValue: number;
    offValue: number;
    dtype: DataType;
}
declare const Pack = "Pack";
declare type PackInputs = TensorInfo[];
interface PackAttrs {
    axis: number;
}
declare const PadV2 = "PadV2";
declare type PadV2Inputs = Pick<NamedTensorInfoMap, 'x'>;
interface PadV2Attrs {
    paddings: Array<[number, number]>;
    constantValue: number;
}
declare const Pool = "Pool";
declare type PoolInputs = Pick<NamedTensorInfoMap, 'input'>;
declare const Pow = "Pow";
declare type PowInputs = BinaryInputs;
declare const Prelu = "Prelu";
declare type PreluInputs = Pick<NamedTensorInfoMap, 'x' | 'alpha'>;
declare const Prod = "Prod";
declare type ProdInputs = Pick<NamedTensorInfoMap, 'x'>;
interface ProdAttrs {
    axis: number | number[];
    keepDims: boolean;
}
declare const RaggedGather = "RaggedGather";
declare type RaggedGatherInputs = {
    paramsNestedSplits: TensorInfo[];
} & Pick<NamedTensorInfoMap, 'paramsDenseValues' | 'indices'>;
interface RaggedGatherAttrs {
    outputRaggedRank: number;
}
declare const RaggedRange = "RaggedRange";
declare type RaggedRangeInputs = Pick<NamedTensorInfoMap, 'starts' | 'limits' | 'deltas'>;
declare const RaggedTensorToTensor = "RaggedTensorToTensor";
declare type RaggedTensorToTensorInputs = Pick<NamedTensorInfoMap, 'shape' | 'values' | 'defaultValue'> & {
    rowPartitionTensors: TensorInfo[];
};
interface RaggedTensorToTensorAttrs {
    rowPartitionTypes: string[];
}
declare const Range = "Range";
interface RangeAttrs {
    start: number;
    stop: number;
    step: number;
    dtype: 'float32' | 'int32';
}
declare const Real = "Real";
declare type RealInputs = Pick<NamedTensorInfoMap, 'input'>;
declare const Reciprocal = "Reciprocal";
declare type ReciprocalInputs = UnaryInputs;
declare const Relu = "Relu";
declare type ReluInputs = Pick<NamedTensorInfoMap, 'x'>;
declare const Reshape = "Reshape";
declare type ReshapeInputs = Pick<NamedTensorInfoMap, 'x'>;
interface ReshapeAttrs {
    shape: number[];
}
declare const ResizeNearestNeighbor = "ResizeNearestNeighbor";
declare type ResizeNearestNeighborInputs = Pick<NamedTensorInfoMap, 'images'>;
interface ResizeNearestNeighborAttrs {
    alignCorners: boolean;
    halfPixelCenters: boolean;
    size: [number, number];
}
declare const ResizeNearestNeighborGrad = "ResizeNearestNeighborGrad";
declare type ResizeNearestNeighborGradInputs = Pick<NamedTensorInfoMap, 'images' | 'dy'>;
declare type ResizeNearestNeighborGradAttrs = ResizeNearestNeighborAttrs;
declare const ResizeBilinear = "ResizeBilinear";
declare type ResizeBilinearInputs = Pick<NamedTensorInfoMap, 'images'>;
interface ResizeBilinearAttrs {
    alignCorners: boolean;
    halfPixelCenters: boolean;
    size: [number, number];
}
declare const ResizeBilinearGrad = "ResizeBilinearGrad";
declare type ResizeBilinearGradInputs = Pick<NamedTensorInfoMap, 'images' | 'dy'>;
declare type ResizeBilinearGradAttrs = ResizeBilinearAttrs;
declare const Relu6 = "Relu6";
declare type Relu6Inputs = Pick<NamedTensorInfoMap, 'x'>;
declare const Reverse = "Reverse";
declare type ReverseInputs = Pick<NamedTensorInfoMap, 'x'>;
interface ReverseAttrs {
    dims: number | number[];
}
declare const Round = "Round";
declare type RoundInputs = UnaryInputs;
declare const Rsqrt = "Rsqrt";
declare type RsqrtInputs = UnaryInputs;
declare const ScatterNd = "ScatterNd";
declare type ScatterNdInputs = Pick<NamedTensorInfoMap, 'indices' | 'updates'>;
interface ScatterNdAttrs {
    shape: number[];
}
declare const SearchSorted = "SearchSorted";
declare type SearchSortedInputs = Pick<NamedTensorInfoMap, 'sortedSequence' | 'values'>;
interface SearchSortedAttrs {
    side: 'left' | 'right';
}
declare const Select = "Select";
declare type SelectInputs = Pick<NamedTensorInfoMap, 'condition' | 't' | 'e'>;
declare const Selu = "Selu";
declare type SeluInputs = Pick<NamedTensorInfoMap, 'x'>;
declare const Slice = "Slice";
declare type SliceInputs = Pick<NamedTensorInfoMap, 'x'>;
interface SliceAttrs {
    begin: number | number[];
    size: number | number[];
}
declare const Sin = "Sin";
declare type SinInputs = UnaryInputs;
declare const Sinh = "Sinh";
declare type SinhInputs = UnaryInputs;
declare const Sign = "Sign";
declare type SignInputs = UnaryInputs;
declare const Sigmoid = "Sigmoid";
declare type SigmoidInputs = UnaryInputs;
declare const Softplus = "Softplus";
declare type SoftplusInputs = UnaryInputs;
declare const Sqrt = "Sqrt";
declare type SqrtInputs = UnaryInputs;
declare const Sum = "Sum";
declare type SumInputs = Pick<NamedTensorInfoMap, 'x'>;
interface SumAttrs {
    axis: number | number[];
    keepDims: boolean;
}
declare const SpaceToBatchND = "SpaceToBatchND";
declare type SpaceToBatchNDInputs = Pick<NamedTensorInfoMap, 'x'>;
interface SpaceToBatchNDAttrs {
    blockShape: number[];
    paddings: number[][];
}
declare const SplitV = "SplitV";
declare type SplitVInputs = Pick<NamedTensorInfoMap, 'x'>;
interface SplitVAttrs {
    numOrSizeSplits: number[] | number;
    axis: number;
}
declare const Softmax = "Softmax";
declare type SoftmaxInputs = Pick<NamedTensorInfoMap, 'logits'>;
interface SoftmaxAttrs {
    dim: number;
}
declare const SparseFillEmptyRows = "SparseFillEmptyRows";
declare type SparseFillEmptyRowsInputs = Pick<NamedTensorInfoMap, 'indices' | 'values' | 'denseShape' | 'defaultValue'>;
declare const SparseReshape = "SparseReshape";
declare type SparseReshapeInputs = Pick<NamedTensorInfoMap, 'inputIndices' | 'inputShape' | 'newShape'>;
declare const SparseSegmentMean = "SparseSegmentMean";
declare type SparseSegmentMeanInputs = Pick<NamedTensorInfoMap, 'data' | 'indices' | 'segmentIds'>;
declare const SparseSegmentSum = "SparseSegmentSum";
declare type SparseSegmentSumInputs = Pick<NamedTensorInfoMap, 'data' | 'indices' | 'segmentIds'>;
declare const SparseToDense = "SparseToDense";
declare type SparseToDenseInputs = Pick<NamedTensorInfoMap, 'sparseIndices' | 'sparseValues' | 'defaultValue'>;
interface SparseToDenseAttrs {
    outputShape: number[];
}
declare const SquaredDifference = "SquaredDifference";
declare type SquaredDifferenceInputs = BinaryInputs;
declare const Square = "Square";
declare type SquareInputs = Pick<NamedTensorInfoMap, 'x'>;
declare const StridedSlice = "StridedSlice";
declare type StridedSliceInputs = Pick<NamedTensorInfoMap, 'x'>;
interface StridedSliceAttrs {
    begin: number[];
    end: number[];
    strides: number[];
    beginMask: number;
    endMask: number;
    ellipsisMask: number;
    newAxisMask: number;
    shrinkAxisMask: number;
}
declare const StringNGrams = "StringNGrams";
declare type StringNGramsInputs = Pick<NamedTensorInfoMap, 'data' | 'dataSplits'>;
interface StringNGramsAttrs {
    separator: string;
    nGramWidths: number[];
    leftPad: string;
    rightPad: string;
    padWidth: number;
    preserveShortSequences: boolean;
}
declare const StringSplit = "StringSplit";
declare type StringSplitInputs = Pick<NamedTensorInfoMap, 'input' | 'delimiter'>;
interface StringSplitAttrs {
    skipEmpty: boolean;
}
declare const StringToHashBucketFast = "StringToHashBucketFast";
declare type StringToHashBucketFastInputs = Pick<NamedTensorInfoMap, 'input'>;
interface StringToHashBucketFastAttrs {
    numBuckets: number;
}
declare const Sub = "Sub";
declare type SubInputs = BinaryInputs;
declare const Tan = "Tan";
declare type TanInputs = UnaryInputs;
declare const Tanh = "Tanh";
declare type TanhInputs = UnaryInputs;
declare const Tile = "Tile";
declare type TileInputs = Pick<NamedTensorInfoMap, 'x'>;
interface TileAttrs {
    reps: number[];
}
declare const TopK = "TopK";
declare type TopKInputs = Pick<NamedTensorInfoMap, 'x'>;
interface TopKAttrs {
    k: number;
    sorted: boolean;
}
declare const Transform = "Transform";
declare type TransformInputs = Pick<NamedTensorInfoMap, 'image' | 'transforms'>;
interface TransformAttrs {
    interpolation: 'nearest' | 'bilinear';
    fillMode: 'constant' | 'reflect' | 'wrap' | 'nearest';
    fillValue: number;
    outputShape?: [number, number];
}
declare const Transpose = "Transpose";
declare type TransposeInputs = Pick<NamedTensorInfoMap, 'x'>;
interface TransposeAttrs {
    perm: number[];
}
declare const Unique = "Unique";
declare type UniqueInputs = Pick<NamedTensorInfoMap, 'x'>;
interface UniqueAttrs {
    axis: number;
}
declare type UnaryInputs = Pick<NamedTensorInfoMap, 'x'>;
declare const Unpack = "Unpack";
declare type UnpackInputs = Pick<NamedTensorInfoMap, 'value'>;
interface UnpackAttrs {
    axis: number;
}
declare const UnsortedSegmentSum = "UnsortedSegmentSum";
declare type UnsortedSegmentSumInputs = Pick<NamedTensorInfoMap, 'x' | 'segmentIds'>;
interface UnsortedSegmentSumAttrs {
    numSegments: number;
}
declare const UpperBound = "UpperBound";
declare type UpperBoundInputs = Pick<NamedTensorInfoMap, 'sortedSequence' | 'values'>;
declare const ZerosLike = "ZerosLike";
declare type ZerosLikeInputs = UnaryInputs;
/**
 * TensorFlow.js-only kernels
 */
declare const Step = "Step";
declare type StepInputs = UnaryInputs;
interface StepAttrs {
    alpha: number;
}
declare const FromPixels = "FromPixels";
interface FromPixelsInputs {
    pixels: PixelData | ImageData | HTMLImageElement | HTMLCanvasElement | HTMLVideoElement | ImageBitmap;
}
interface FromPixelsAttrs {
    numChannels: number;
}
declare const RotateWithOffset = "RotateWithOffset";
declare type RotateWithOffsetInputs = Pick<NamedTensorInfoMap, 'image'>;
interface RotateWithOffsetAttrs {
    radians: number;
    fillValue: number | [number, number, number];
    center: number | [number, number];
}
declare const _FusedMatMul = "_FusedMatMul";
interface _FusedMatMulInputs extends NamedTensorInfoMap {
    a: TensorInfo;
    b: TensorInfo;
    bias?: TensorInfo;
    preluActivationWeights?: TensorInfo;
}
interface _FusedMatMulAttrs {
    transposeA: boolean;
    transposeB: boolean;
    activation: Activation;
    leakyreluAlpha?: number;
}
declare const FusedConv2D = "FusedConv2D";
interface FusedConv2DInputs extends NamedTensorInfoMap {
    x: TensorInfo;
    filter: TensorInfo;
    bias?: TensorInfo;
    preluActivationWeights?: TensorInfo;
}
interface FusedConv2DAttrs {
    strides: [number, number] | number;
    pad: 'valid' | 'same' | number | ExplicitPadding;
    dataFormat: 'NHWC' | 'NCHW';
    dilations: [number, number] | number;
    dimRoundingMode: 'floor' | 'round' | 'ceil';
    activation: Activation;
    leakyreluAlpha?: number;
}
declare const FusedDepthwiseConv2D = "FusedDepthwiseConv2D";
interface FusedDepthwiseConv2DInputs extends NamedTensorInfoMap {
    x: TensorInfo;
    filter: TensorInfo;
    bias?: TensorInfo;
    preluActivationWeights?: TensorInfo;
}
interface FusedDepthwiseConv2DAttrs {
    strides: [number, number] | number;
    pad: 'valid' | 'same' | number | ExplicitPadding;
    dataFormat: 'NHWC' | 'NCHW';
    dilations: [number, number] | number;
    dimRoundingMode: 'floor' | 'round' | 'ceil';
    activation: Activation;
    leakyreluAlpha?: number;
}
/// <amd-module name="@tensorflow/tfjs-core/dist/kernel_registry" />
declare type AttributeValue = number | number[] | boolean | boolean[] | string | string[] | NamedAttrMap;
/** These are extra non-tensor/primitive params passed to kernel functions. */
declare type Attribute = AttributeValue | RecursiveArray<AttributeValue>;
/** Specifies the code to run when executing a kernel. */
declare type KernelFunc = (params: {
    inputs: NamedTensorInfoMap;
    backend: {};
    attrs?: NamedAttrMap;
}) => TensorInfo | TensorInfo[];
/** The function to run when computing a gradient during backprop. */
declare type GradFunc = (dy: Tensor | Tensor[], saved: Tensor[], attrs: NamedAttrMap) => NamedGradientMap;
/** Function that gets called after the backend initializes. */
declare type KernelSetupFunc = (backend: {}) => void;
/** Function that gets called right before the backend is disposed. */
declare type KernelDisposeFunc = KernelSetupFunc;
/** Config object for registering a kernel in the global registry. */
interface KernelConfig {
    kernelName: string;
    backendName: string;
    kernelFunc: KernelFunc;
    setupFunc?: KernelSetupFunc;
    disposeFunc?: KernelDisposeFunc;
}
/** Config object for registering a gradient in the global registry. */
interface GradConfig {
    kernelName: string;
    inputsToSave?: string[];
    saveAllInputs?: boolean;
    outputsToSave?: boolean[];
    gradFunc: GradFunc;
}
interface NamedTensorInfoMap {
    [name: string]: TensorInfo | undefined;
}
interface NamedAttrMap {
    [name: string]: Attribute;
}
/**
 * Returns the kernel function (code) associated with the provided names.
 *
 * @param kernelName The official name of the kernel.
 * @param backendName The official name of the backend.
 */
declare function getKernel(kernelName: string, backendName: string): KernelConfig;
/**
 * Returns the registered gradient info associated with the provided kernel.
 * @param kernelName The official TF kernel name.
 */
declare function getGradient(kernelName: string): GradConfig;
declare function getKernelsForBackend(backendName: string): KernelConfig[];
/**
 * Registers the function (forward pass) for the kernel in a global registry.
 *
 * @param config A config object with the following properties:
 * - `kernelName` The official name of the kernel.
 * - `backendName` The official name of the backend.
 * - `kernelFunc` The function to run during the forward pass of the kernel.
 * - `setupFunc` Optional. Gets called once, after the backend initializes.
 * - `disposeFunc` Optional. Gets called once, right before the backend is
 * disposed.
 */
declare function registerKernel(config: KernelConfig): void;
/**
 * Registers a gradient function for a given kernel in the global registry,
 * to be used during the back-propagation of that kernel.
 *
 * @param config An object with the following properties:
 * - `kernelName` The name of the kernel that the gradient function is for.
 * - `gradFunc` The function to run during back-propagation.
 */
declare function registerGradient(config: GradConfig): void;
/**
 * Removes the kernel function from the registry.
 *
 * @param kernelName The official name of the kernel.
 * @param backendName The official name of the backend.
 *
 */
declare function unregisterKernel(kernelName: string, backendName: string): void;
/** Removes the registered gradient from the global registry. */
declare function unregisterGradient(kernelName: string): void;
/**
 * Finds kernels that have already been registered to a backend and re-registers
 * them for a new backend. Useful for registering custom backends.
 * @param registeredBackendName Already registered backend.
 * @param newBackendName New backend.
 */
declare function copyRegisteredKernels(registeredBackendName: string, newBackendName: string): void;
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/kernel_registry_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-layers/dist/keras_format/layers/layer_serialization" />
declare type LayerSerialization = AdvancedActivationLayerSerialization | ConvolutionalDepthwiseLayerSerialization | ConvolutionalLayerSerialization | CoreLayerSerialization | EmbeddingLayerSerialization | MergeLayerSerialization | NormalizationLayerSerialization | PaddingLayerSerialization | PoolingLayerSerialization | RecurrentLayerSerialization | InputLayerSerialization;
declare type LayerClassName = LayerSerialization['class_name'];
/**
 * A string array of valid Layer class names.
 *
 * This is guaranteed to match the `LayerClassName` union type.
 */
declare const layerClassNames: LayerClassName[];

/// <amd-module name="@tensorflow/tfjs-layers/dist/utils/layer_utils" />
/**
 * Print the summary of a LayersModel object.
 *
 * @param model tf.LayersModel instance.
 * @param lineLength Total length of printed lines. Set this to adapt to the
 *   display to different terminal or console sizes.
 * @param positions Relative or absolute positions of log elements in each
 *   line. Each number corresponds to right-most (i.e., ending) position of a
 *   column.
 *   If not provided, defaults to `[0.45, 0.85, 1]` for sequential-like
 *   models and `[0.33, 0.55, 0.67, 1]` for non-sequential like models.
 * @param printFn Print function to use.
 *   It will be called on each line of the summary. You can provide a custom
 *   function in order to capture the string summary. Defaults to `console.log`.
 */
declare function printSummary(model: Container, lineLength?: number, positions?: number[], printFn?: (message?: any, ...optionalParams: any[]) => void): void;

/// <amd-module name="@tensorflow/tfjs-data/dist/iterators/lazy_iterator" />
/**
 * A nested structure of LazyIterators, used as the input to zip().
 */
declare type IteratorContainer = Container<LazyIterator<tf.TensorContainer>>;
/**
 * Create a `LazyIterator` from an array of items.
 */
declare function iteratorFromItems<T>(items: T[]): LazyIterator<T>;
/**
 * Create a `LazyIterator` of incrementing integers.
 */
declare function iteratorFromIncrementing(start: number): LazyIterator<number>;
/**
 * Create a `LazyIterator` from a function.
 *
 * ```js
 * let i = -1;
 * const func = () =>
 *    ++i < 5 ? {value: i, done: false} : {value: null, done: true};
 * const iter = tf.data.iteratorFromFunction(func);
 * await iter.forEachAsync(e => console.log(e));
 * ```
 *
 * @param func A function that produces data on each call.
 */
declare function iteratorFromFunction<T>(func: () => IteratorResult<T> | Promise<IteratorResult<T>>): LazyIterator<T>;
/**
 * Create a `LazyIterator` by concatenating underlying streams, which are
 * themselves provided as a stream.
 *
 * This can also be thought of as a "stream flatten" operation.
 *
 * @param baseIterators A stream of streams to be concatenated.
 * @param baseErrorHandler An optional function that can intercept `Error`s
 *   raised during a `next()` call on the base stream.  This function can decide
 *   whether the error should be propagated, whether the error should be
 *   ignored, or whether the base stream should be terminated.
 */
declare function iteratorFromConcatenated<T>(baseIterators: LazyIterator<LazyIterator<T>>, baseErrorHandler?: (e: Error) => boolean): LazyIterator<T>;
/**
 * Create a `LazyIterator` by concatenating streams produced by calling a
 * stream-generating function a given number of times.
 *
 * Since a `LazyIterator` is read-once, it cannot be repeated, but this
 * function can be used to achieve a similar effect:
 *
 *   LazyIterator.ofConcatenatedFunction(() => new MyIterator(), 6);
 *
 * @param iteratorFunc: A function that produces a new stream on each call.
 * @param count: The number of times to call the function.
 * @param baseErrorHandler An optional function that can intercept `Error`s
 *   raised during a `next()` call on the base stream.  This function can decide
 *   whether the error should be propagated, whether the error should be
 *   ignored, or whether the base stream should be terminated.
 */
declare function iteratorFromConcatenatedFunction<T>(iteratorFunc: () => IteratorResult<LazyIterator<T>>, count: number, baseErrorHandler?: (e: Error) => boolean): LazyIterator<T>;
/**
 * Create a `LazyIterator` by zipping together an array, dict, or nested
 * structure of `LazyIterator`s (and perhaps additional constants).
 *
 * The underlying streams must provide elements in a consistent order such
 * that they correspond.
 *
 * Typically, the underlying streams should have the same number of
 * elements. If they do not, the behavior is determined by the
 * `mismatchMode` argument.
 *
 * The nested structure of the `iterators` argument determines the
 * structure of elements in the resulting iterator.
 *
 * @param iterators: An array or object containing LazyIterators at the
 * leaves.
 * @param mismatchMode: Determines what to do when one underlying iterator
 * is exhausted before the others.  `ZipMismatchMode.FAIL` (the default)
 * causes an error to be thrown in this case.  `ZipMismatchMode.SHORTEST`
 * causes the zipped iterator to terminate with the furst underlying
 * streams, so elements remaining on the longer streams are ignored.
 * `ZipMismatchMode.LONGEST` causes the zipped stream to continue, filling
 * in nulls for the exhausted streams, until all streams are exhausted.
 */
declare function iteratorFromZipped<O extends tf.TensorContainer>(iterators: IteratorContainer, mismatchMode?: ZipMismatchMode): LazyIterator<O>;
/**
 * An asynchronous iterator, providing lazy access to a potentially
 * unbounded stream of elements.
 *
 * Iterator can be obtained from a dataset:
 * `const iter = await dataset.iterator();`
 */
declare abstract class LazyIterator<T> {
    abstract summary(): string;
    /**
     * Returns a `Promise` for the next element in the stream.
     *
     * When an item can be provided successfully, the return value is
     * `{value:T, done:false}`.
     *
     * Calling next() on a closed stream returns `{value:null, done:true}`.
     */
    abstract next(): Promise<IteratorResult<T>>;
    /**
     * Collect all remaining elements of a bounded stream into an array.
     * Obviously this will succeed only for small streams that fit in memory.
     * Useful for testing.
     *
     * @returns A Promise for an array of stream elements, which will resolve
     *   when the stream is exhausted.
     */
    toArray(): Promise<T[]>;
    /**
     * Collect all elements of this dataset into an array with prefetching 100
     * elements. This is useful for testing, because the prefetch changes the
     * order in which the Promises are resolved along the processing pipeline.
     * This may help expose bugs where results are dependent on the order of
     * Promise resolution rather than on the logical order of the stream (i.e.,
     * due to hidden mutable state).
     *
     * @returns A Promise for an array of stream elements, which will resolve
     *   when the stream is exhausted.
     */
    toArrayForTest(): Promise<T[]>;
    /**
     * Draw items from the stream until it is exhausted.
     *
     * This can be useful when the stream has side effects but no output.  In
     * that case, calling this function guarantees that the stream will be
     * fully processed.
     */
    resolveFully(): Promise<void>;
    /**
     * Draw items from the stream until it is exhausted, or a predicate fails.
     *
     * This can be useful when the stream has side effects but no output.  In
     * that case, calling this function guarantees that the stream will be
     * fully processed.
     */
    resolveWhile(predicate: (r: T) => boolean): Promise<void>;
    /**
     * Handles errors thrown on this stream using a provided handler function.
     *
     * @param handler A function that handles any `Error` thrown during a `next()`
     *   call and returns true if the stream should continue (dropping the failed
     *   call) or false if the stream should quietly terminate.  If the handler
     *   itself throws (or rethrows) an `Error`, that will be propagated.
     *
     * @returns A `LazyIterator` of elements passed through from upstream,
     *   possibly filtering or terminating on upstream `next()` calls that
     *   throw an `Error`.
     */
    handleErrors(handler: (error: Error) => boolean): LazyIterator<T>;
    /**
     * Filters this stream according to `predicate`.
     *
     * @param predicate A function mapping a stream element to a boolean or a
     * `Promise` for one.
     *
     * @returns A `LazyIterator` of elements for which the predicate was true.
     */
    filter(predicate: (value: T) => boolean): LazyIterator<T>;
    /**
     * Maps this stream through a 1-to-1 transform.
     *
     * @param transform A function mapping a stream element to a transformed
     *   element.
     *
     * @returns A `LazyIterator` of transformed elements.
     */
    map<O>(transform: (value: T) => O): LazyIterator<O>;
    /**
     * Maps this stream through an async 1-to-1 transform.
     *
     * @param transform A function mapping a stream element to a `Promise` for a
     *   transformed stream element.
     *
     * @returns A `LazyIterator` of transformed elements.
     */
    mapAsync<O>(transform: (value: T) => Promise<O>): LazyIterator<O>;
    /**
     * Maps this stream through a 1-to-1 transform, forcing serial execution.
     *
     * @param transform A function mapping a stream element to a transformed
     *   element.
     *
     * @returns A `LazyIterator` of transformed elements.
     */
    serialMapAsync<O>(transform: (value: T) => Promise<O>): LazyIterator<O>;
    /**
     * Maps this stream through a 1-to-many transform.
     *
     * @param transform A function mapping a stream element to an array of
     *   transformed elements.
     *
     * @returns A `DataStream` of transformed elements.
     */
    flatmap<O>(transform: (value: T) => O[]): LazyIterator<O>;
    /**
     * Apply a function to every element of the stream.
     *
     * @param f A function to apply to each stream element.
     */
    forEachAsync(f: (value: T) => void): Promise<void>;
    /**
     * Apply a function to every element of the stream, forcing serial execution.
     *
     * @param f A function to apply to each stream element.  Should return 'true'
     *   to indicate that the stream should continue, or 'false' to cause it to
     *   terminate.
     */
    serialForEach(f: (value: T) => Promise<boolean>): Promise<void>;
    /**
     * Groups elements into batches, represented as arrays of elements.
     *
     * We can think of the elements of this iterator as 'rows' (even if they are
     * nested structures).  By the same token, consecutive values for a given
     * key within the elements form a 'column'.  This matches the usual sense of
     * 'row' and 'column' when processing tabular data (e.g., parsing a CSV).
     *
     * Thus, "Row-major" means that the resulting batch is simply a collection of
     * rows: `[row1, row2, row3, ...]`.  This is contrast to the column-major
     * form, which is needed for vectorized computation.
     *
     * @param batchSize The number of elements desired per batch.
     * @param smallLastBatch Whether to emit the final batch when it has fewer
     *   than batchSize elements. Default true.
     * @returns A `LazyIterator` of batches of elements, represented as arrays
     *   of the original element type.
     */
    rowMajorBatch(batchSize: number, smallLastBatch?: boolean): LazyIterator<T[]>;
    /**
     * Groups elements into batches, represented in column-major form.
     *
     * We can think of the elements of this iterator as 'rows' (even if they are
     * nested structures).  By the same token, consecutive values for a given
     * key within the elements form a 'column'.  This matches the usual sense of
     * 'row' and 'column' when processing tabular data (e.g., parsing a CSV).
     *
     * Thus, "column-major" means that the resulting batch is a (potentially
     * nested) structure representing the columns.  Each column entry, then,
     * contains a collection of the values found in that column for a range of
     * input elements.  This representation allows for vectorized computation, in
     * contrast to the row-major form.
     *
     * The inputs should all have the same nested structure (i.e., of arrays and
     * dicts).  The result is a single object with the same nested structure,
     * where the leaves are arrays collecting the values of the inputs at that
     * location (or, optionally, the result of a custom function applied to those
     * arrays).
     *
     * @param batchSize The number of elements desired per batch.
     * @param smallLastBatch Whether to emit the final batch when it has fewer
     *   than batchSize elements. Default true.
     * @param zipFn: (optional) A function that expects an array of elements at a
     *   single node of the object tree, and returns a `DeepMapResult`.  The
     *   `DeepMapResult` either provides a result value for that node (i.e.,
     *   representing the subtree), or indicates that the node should be processed
     *   recursively.  The default zipFn recurses as far as possible and places
     *   arrays at the leaves.
     * @returns A `LazyIterator` of batches of elements, represented as an object
     *   with collections at the leaves.
     */
    columnMajorBatch(batchSize: number, smallLastBatch?: boolean, zipFn?: (xs: any[]) => DeepMapResult): LazyIterator<tf.TensorContainer>;
    /**
     * Concatenate this `LazyIterator` with another.
     *
     * @param iterator A `LazyIterator` to be concatenated onto this one.
     * @param baseErrorHandler An optional function that can intercept `Error`s
     *   raised during a `next()` call on the base stream.  This function can
     *   decide whether the error should be propagated, whether the error should
     *   be ignored, or whether the base stream should be terminated.
     * @returns A `LazyIterator`.
     */
    concatenate(iterator: LazyIterator<T>, baseErrorHandler?: (e: Error) => boolean): LazyIterator<T>;
    /**
     * Limits this stream to return at most `count` items.
     *
     * @param count The maximum number of items to provide from the stream. If
     * a negative or undefined value is given, the entire stream is returned
     *   unaltered.
     */
    take(count: number): LazyIterator<T>;
    /**
     * Skips the first `count` items in this stream.
     *
     * @param count The number of items to skip.  If a negative or undefined
     * value is given, the entire stream is returned unaltered.
     */
    skip(count: number): LazyIterator<T>;
    /**
     * Prefetch the first `bufferSize` items in this stream.
     *
     * Note this prefetches Promises, but makes no guarantees about when those
     * Promises resolve.
     *
     * @param bufferSize: An integer specifying the number of elements to be
     *   prefetched.
     */
    prefetch(bufferSize: number): LazyIterator<T>;
    /**
     * Randomly shuffles the elements of this stream.
     *
     * @param bufferSize: An integer specifying the number of elements from
     * this stream from which the new stream will sample.
     * @param seed: (Optional.) An integer specifying the random seed that
     * will be used to create the distribution.
     */
    shuffle(windowSize: number, seed?: string): LazyIterator<T>;
    /**
     * Force an iterator to execute serially: each next() call will await the
     * prior one, so that they cannot execute concurrently.
     */
    serial(): LazyIterator<T>;
}
/**
 * A base class for transforming streams that operate by maintaining an
 * output queue of elements that are ready to return via next().  This is
 * commonly required when the transformation is 1-to-many:  A call to next()
 * may trigger a call to the underlying stream, which will produce many
 * mapped elements of this stream-- of which we need to return only one, so
 * we have to queue the rest.
 */
declare abstract class OneToManyIterator<T> extends LazyIterator<T> {
    private lastRead;
    protected outputQueue: RingBuffer<T>;
    constructor();
    next(): Promise<IteratorResult<T>>;
    /**
     * Read one or more chunks from upstream and process them, possibly
     * reading or writing a carryover, and adding processed items to the
     * output queue.  Note it's possible that no items are added to the queue
     * on a given pump() call, even if the upstream stream is not closed
     * (e.g., because items are filtered).
     *
     * @return `true` if any action was taken, i.e. fetching items from the
     *   upstream source OR adding items to the output queue.  `false` if the
     *   upstream source is exhausted AND nothing was added to the queue
     * (i.e., any remaining carryover).
     */
    protected abstract pump(): Promise<boolean>;
    serialNext(): Promise<IteratorResult<T>>;
}
/**
 * Provides a `LazyIterator` that concatenates a stream of underlying
 * streams.
 *
 * Doing this in a concurrency-safe way requires some trickery.  In
 * particular, we want this stream to return the elements from the
 * underlying streams in the correct order according to when next() was
 * called, even if the resulting Promises resolve in a different order.
 */
declare class ChainedIterator<T> extends LazyIterator<T> {
    private readonly baseErrorHandler?;
    private lastRead;
    private iterator;
    private moreIterators;
    constructor(iterators: LazyIterator<LazyIterator<T>>, baseErrorHandler?: (e: Error) => boolean);
    summary(): string;
    next(): Promise<IteratorResult<T>>;
    private readFromChain;
}
declare enum ZipMismatchMode {
    FAIL = 0,
    SHORTEST = 1,
    LONGEST = 2
}
/**
 * A stream that prefetches a given number of items from an upstream source,
 * returning them in FIFO order.
 *
 * Note this prefetches Promises, but makes no guarantees about when those
 * Promises resolve.
 */
declare class PrefetchIterator<T> extends LazyIterator<T> {
    protected upstream: LazyIterator<T>;
    protected bufferSize: number;
    protected buffer: RingBuffer<Promise<IteratorResult<T>>>;
    constructor(upstream: LazyIterator<T>, bufferSize: number);
    summary(): string;
    /**
     * Refill the prefetch buffer.  Returns only after the buffer is full, or
     * the upstream source is exhausted.
     */
    protected refill(): void;
    next(): Promise<IteratorResult<T>>;
}
/**
 * A stream that performs a sliding-window random shuffle on an upstream
 * source. This is like a `PrefetchIterator` except that the items are
 * returned in randomized order.  Mixing naturally improves as the buffer
 * size increases.
 */
declare class ShuffleIterator<T> extends PrefetchIterator<T> {
    protected upstream: LazyIterator<T>;
    protected windowSize: number;
    private readonly random;
    private lastRead;
    private upstreamExhausted;
    constructor(upstream: LazyIterator<T>, windowSize: number, seed?: string);
    next(): Promise<IteratorResult<T>>;
    private randomInt;
    protected chooseIndex(): number;
    serialNext(): Promise<IteratorResult<T>>;
}
/// <amd-module name="@tensorflow/tfjs-core/dist/gradients/LeakyRelu_grad" />
declare const leakyReluGradConfig: GradConfig;
/// <amd-module name="@tensorflow/tfjs-core/dist/public/chained_ops/leaky_relu" />
declare module '../../tensor' {
    interface Tensor<R extends Rank = Rank> {
        leakyRelu<T extends Tensor>(alpha: number): T;
    }
}

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/leaky_relu_test" />
{ };
/// <amd-module name="@tensorflow/tfjs-core/dist/ops/less" />
/**
 * Returns the truth value of (a < b) element-wise. Supports broadcasting.
 *
 * ```js
 * const a = tf.tensor1d([1, 2, 3]);
 * const b = tf.tensor1d([2, 2, 2]);
 *
 * a.less(b).print();
 * ```
 * @param a The first input tensor.
 * @param b The second input tensor. Must have the same dtype as `a`.
 *
 * @doc {heading: 'Operations', subheading: 'Logical'}
 */
declare function less_<T extends Tensor>(a: Tensor | TensorLike, b: Tensor | TensorLike): T;
declare const less: typeof less_;
{ };
/// <amd-module name="@tensorflow/tfjs-core/dist/ops/less_equal" />
/**
 * Returns the truth value of (a <= b) element-wise. Supports broadcasting.
 *
 * ```js
 * const a = tf.tensor1d([1, 2, 3]);
 * const b = tf.tensor1d([2, 2, 2]);
 *
 * a.lessEqual(b).print();
 * ```
 *
 * @param a The first input tensor.
 * @param b The second input tensor. Must have the same dtype as `a`.
 *
 * @doc {heading: 'Operations', subheading: 'Logical'}
 */
declare function lessEqual_<T extends Tensor>(a: Tensor | TensorLike, b: Tensor | TensorLike): T;
declare const lessEqual: typeof lessEqual_;
{ };
/// <amd-module name="@tensorflow/tfjs-core/dist/ops/less_equal_test" />
{ };
/// <amd-module name="@tensorflow/tfjs-core/dist/ops/less_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/linspace" />
/**
 * Return an evenly spaced sequence of numbers over the given interval.
 *
 * ```js
 * tf.linspace(0, 9, 10).print();
 * ```
 * @param start The start value of the sequence.
 * @param stop The end value of the sequence.
 * @param num The number of values to generate.
 *
 * @doc {heading: 'Tensors', subheading: 'Creation'}
 */
declare function linspace(start: number, stop: number, num: number): Tensor1D;

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/linspace_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/local_response_normalization" />
/**
 * Normalizes the activation of a local neighborhood across or within
 * channels.
 *
 * @param x The input tensor. The 4-D input tensor is treated as a 3-D array
 *     of 1D vectors (along the last dimension), and each vector is
 *     normalized independently.
 * @param depthRadius The number of adjacent channels in the 1D normalization
 *     window.
 * @param bias A constant bias term for the basis.
 * @param alpha A scale factor, usually positive.
 * @param beta An exponent.
 *
 * @doc {heading: 'Operations', subheading: 'Normalization'}
 */
declare function localResponseNormalization_<T extends Tensor3D | Tensor4D>(x: T | TensorLike, depthRadius?: number, bias?: number, alpha?: number, beta?: number): T;
declare const localResponseNormalization: typeof localResponseNormalization_;
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/local_response_normalization_backprop" />
declare function localResponseNormalizationBackprop_<T extends Tensor4D>(x: T, y: T, dy: T, depthRadius?: number, bias?: number, alpha?: number, beta?: number): T;
declare const localResponseNormalizationBackprop: typeof localResponseNormalizationBackprop_;
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/local_response_normalization_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/io/local_storage" />
/**
 * Purge all tensorflow.js-saved model artifacts from local storage.
 *
 * @returns Paths of the models purged.
 */
declare function purgeLocalStorageArtifacts(): string[];
declare type LocalStorageKeys = {
    /** Key of the localStorage entry storing `ModelArtifactsInfo`. */
    info: string;
    /**
     * Key of the localStorage entry storing the 'modelTopology' key of
     * `model.json`
     */
    topology: string;
    /**
     * Key of the localStorage entry storing the `weightsManifest.weights` entries
     * of `model.json`
     */
    weightSpecs: string;
    /** Key of the localStorage entry storing the weight data in Base64 */
    weightData: string;
    /**
     * Key of the localStorage entry storing the remaining fields of `model.json`
     * @see {@link ModelMetadata}
     */
    modelMetadata: string;
};
/**
 * IOHandler subclass: Browser Local Storage.
 *
 * See the doc string to `browserLocalStorage` for more details.
 */
declare class BrowserLocalStorage implements IOHandler {
    protected readonly LS: Storage;
    protected readonly modelPath: string;
    protected readonly keys: LocalStorageKeys;
    static readonly URL_SCHEME = "localstorage://";
    constructor(modelPath: string);
    /**
     * Save model artifacts to browser local storage.
     *
     * See the documentation to `browserLocalStorage` for details on the saved
     * artifacts.
     *
     * @param modelArtifacts The model artifacts to be stored.
     * @returns An instance of SaveResult.
     */
    save(modelArtifacts: ModelArtifacts): Promise<SaveResult>;
    /**
     * Load a model from local storage.
     *
     * See the documentation to `browserLocalStorage` for details on the saved
     * artifacts.
     *
     * @returns The loaded model (if loading succeeds).
     */
    load(): Promise<ModelArtifacts>;
}
declare const localStorageRouter: IORouter;
/**
 * Factory function for local storage IOHandler.
 *
 * This `IOHandler` supports both `save` and `load`.
 *
 * For each model's saved artifacts, four items are saved to local storage.
 *   - `${PATH_SEPARATOR}/${modelPath}/info`: Contains meta-info about the
 *     model, such as date saved, type of the topology, size in bytes, etc.
 *   - `${PATH_SEPARATOR}/${modelPath}/topology`: Model topology. For Keras-
 *     style models, this is a stringized JSON.
 *   - `${PATH_SEPARATOR}/${modelPath}/weight_specs`: Weight specs of the
 *     model, can be used to decode the saved binary weight values (see
 *     item below).
 *   - `${PATH_SEPARATOR}/${modelPath}/weight_data`: Concatenated binary
 *     weight values, stored as a base64-encoded string.
 *
 * Saving may throw an `Error` if the total size of the artifacts exceed the
 * browser-specific quota.
 *
 * @param modelPath A unique identifier for the model to be saved. Must be a
 *   non-empty string.
 * @returns An instance of `IOHandler`, which can be used with, e.g.,
 *   `tf.Model.save`.
 */
declare function browserLocalStorage(modelPath: string): IOHandler;
declare class BrowserLocalStorageManager implements ModelStoreManager {
    private readonly LS;
    constructor();
    listModels(): Promise<{
        [path: string]: ModelArtifactsInfo;
    }>;
    removeModel(path: string): Promise<ModelArtifactsInfo>;
}
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/io/local_storage_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/log" />
/**
 * Computes natural logarithm of the input `tf.Tensor` element-wise: `ln(x)`
 *
 * ```js
 * const x = tf.tensor1d([1, 2, Math.E]);
 *
 * x.log().print();  // or tf.log(x)
 * ```
 * @param x The input tensor.
 *
 * @doc {heading: 'Operations', subheading: 'Basic math'}
 */
declare function log_<T extends Tensor>(x: T | TensorLike): T;
declare const log: typeof log_;
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/log1p" />
/**
 * Computes natural logarithm of the input `tf.Tensor` plus one
 * element-wise: `ln(1 + x)`
 *
 * ```js
 * const x = tf.tensor1d([1, 2, Math.E - 1]);
 *
 * x.log1p().print();  // or tf.log1p(x)
 * ```
 * @param x The input tensor.
 *
 * @doc {heading: 'Operations', subheading: 'Basic math'}
 */
declare function log1p_<T extends Tensor>(x: T | TensorLike): T;
declare const log1p: typeof log1p_;
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/gradients/Log1p_grad" />
declare const log1pGradConfig: GradConfig;

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/log1p_test" />
{ };
/// <amd-module name="@tensorflow/tfjs-core/dist/ops/logical_and" />
/**
 * Returns the truth value of `a AND b` element-wise. Supports broadcasting.
 *
 * ```js
 * const a = tf.tensor1d([false, false, true, true], 'bool');
 * const b = tf.tensor1d([false, true, false, true], 'bool');
 *
 * a.logicalAnd(b).print();
 * ```
 *
 * @param a The first input tensor. Must be of dtype bool.
 * @param b The second input tensor. Must be of dtype bool.
 *
 * @doc {heading: 'Operations', subheading: 'Logical'}
 */
declare function logicalAnd_<T extends Tensor>(a: Tensor | TensorLike, b: Tensor | TensorLike): T;
declare const logicalAnd: typeof logicalAnd_;
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/logical_and_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/logical_not" />
/**
 * Returns the truth value of `NOT x` element-wise.
 *
 * ```js
 * const a = tf.tensor1d([false, true], 'bool');
 *
 * a.logicalNot().print();
 * ```
 *
 * @param x The input tensor. Must be of dtype 'bool'.
 *
 * @doc {heading: 'Operations', subheading: 'Logical'}
 */
declare function logicalNot_<T extends Tensor>(x: T | TensorLike): T;
declare const logicalNot: typeof logicalNot_;
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/logical_not_test" />
{ };
/// <amd-module name="@tensorflow/tfjs-core/dist/public/chained_ops/logical_or" />
declare module '../../tensor' {
    interface Tensor<R extends Rank = Rank> {
        logicalOr<T extends Tensor>(b: Tensor | TensorLike): T;
    }
}
/// <amd-module name="@tensorflow/tfjs-core/dist/ops/logical_or_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/logical_xor" />
/**
 * Returns the truth value of `a XOR b` element-wise. Supports broadcasting.
 *
 * ```js
 * const a = tf.tensor1d([false, false, true, true], 'bool');
 * const b = tf.tensor1d([false, true, false, true], 'bool');
 *
 * a.logicalXor(b).print();
 * ```
 *
 * @param a The first input tensor. Must be of dtype bool.
 * @param b The second input tensor. Must be of dtype bool.
 *
 * @doc {heading: 'Operations', subheading: 'Logical'}
 */
declare function logicalXor_<T extends Tensor>(a: Tensor | TensorLike, b: Tensor | TensorLike): T;
declare const logicalXor: typeof logicalXor_;
{ };
/// <amd-module name="@tensorflow/tfjs-core/dist/ops/logical_xor_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-layers/dist/logs" />
/**
 * Logs in which values can be either numbers or Tensors (Scalars).
 *
 * Used internally.
 */
declare type UnresolvedLogs = {
    [key: string]: number | Scalar;
};
/**
 * Turn any Scalar values in a Logs object into actual number values.
 *
 * @param logs The `Logs` object to be resolved in place.
 */
declare function resolveScalarsInLogs(logs: UnresolvedLogs): Promise<void>;
/**
 * Dispose all Tensors in an UnresolvedLogs object.
 *
 * @param logs An `UnresolvedLogs` object potentially containing `tf.Tensor`s in
 *   places where the values can be `tf.Tensor` or `number`.
 */
declare function disposeTensorsInLogs(logs: UnresolvedLogs): void;
/**
 * Logs in which values can only be numbers.
 *
 * Used when calling client-provided custom callbacks.
 */
declare type Logs = {
    [key: string]: number;
};

/// <amd-module name="@tensorflow/tfjs-core/dist/gradients/LogSoftmax_grad" />
declare const logSoftmaxGradConfig: GradConfig;

/// <amd-module name="@tensorflow/tfjs-core/dist/gradients/Log_grad" />
declare const logGradConfig: GradConfig;

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/losses/log_loss" />
/**
 * Computes the log loss between two tensors.
 *
 * @param labels The ground truth output tensor, same dimensions as
 *    'predictions'.
 * @param predictions The predicted outputs.
 * @param weights Tensor whose rank is either 0, or the same rank as
 *    `labels`, and must be broadcastable to `labels` (i.e., all dimensions
 *    must be either `1`, or the same as the corresponding `losses`
 *    dimension).
 * @param epsilon A small increment to avoid taking log of zero
 * @param reduction Type of reduction to apply to loss. Should be of type
 *    `Reduction`
 *
 * @doc {heading: 'Training', subheading: 'Losses', namespace: 'losses'}
 */
declare function logLoss_<T extends Tensor, O extends Tensor>(labels: T | TensorLike, predictions: T | TensorLike, weights?: Tensor | TensorLike, epsilon?: number, reduction?: Reduction): O;
declare const logLoss: typeof logLoss_;
{ };
/// <amd-module name="@tensorflow/tfjs-core/dist/ops/losses/log_loss_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/log_sigmoid" />
/**
 * Computes log sigmoid of the input `tf.Tensor` element-wise:
 * `logSigmoid(x)`. For numerical stability, we use `-tf.softplus(-x)`.
 *
 * ```js
 * const x = tf.tensor1d([0, 1, -1, .7]);
 *
 * x.logSigmoid().print();  // or tf.logSigmoid(x)
 * ```
 * @param x The input tensor.
 *
 * @doc {heading: 'Operations', subheading: 'Basic math'}
 */
declare function logSigmoid_<T extends Tensor>(x: T | TensorLike): T;
declare const logSigmoid: typeof logSigmoid_;
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/log_sigmoid_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/public/chained_ops/log_softmax" />
declare module '../../tensor' {
    interface Tensor<R extends Rank = Rank> {
        logSoftmax<T extends Tensor>(this: T, axis?: number): T;
    }
}

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/log_softmax_test" />
{ };
/// <amd-module name="@tensorflow/tfjs-core/dist/public/chained_ops/log_sum_exp" />
declare module '../../tensor' {
    interface Tensor<R extends Rank = Rank> {
        logSumExp<T extends Tensor>(this: T, axis?: number | number[], keepDims?: boolean): T;
    }
}

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/log_sum_exp_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/log_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-layers/dist/losses" />
/**
 * Normalizes a tensor wrt the L2 norm alongside the specified axis.
 * @param x
 * @param axis Axis along which to perform normalization.
 */
declare function l2Normalize(x: Tensor, axis?: number): Tensor;
declare function meanSquaredError(yTrue: Tensor, yPred: Tensor): Tensor;
declare function meanAbsoluteError(yTrue: Tensor, yPred: Tensor): Tensor;
declare function meanAbsolutePercentageError(yTrue: Tensor, yPred: Tensor): Tensor;
declare function meanSquaredLogarithmicError(yTrue: Tensor, yPred: Tensor): Tensor;
declare function squaredHinge(yTrue: Tensor, yPred: Tensor): Tensor;
declare function hinge(yTrue: Tensor, yPred: Tensor): Tensor;
declare function categoricalHinge(yTrue: Tensor, yPred: Tensor): Tensor;
/**
 * Logarithm of the hyperbolic cosine of the prediction error.
 *
 * `log(cosh(x))` is approximately equal to `(x ** 2) / 2` for small `x` and
 * to `abs(x) - log(2)` for large `x`. This means that 'logcosh' works mostly
 * like the mean squared error, but will not be so strongly affected by the
 * occasional wildly incorrect prediction.
 */
declare function logcosh(yTrue: Tensor, yPred: Tensor): Tensor;
declare function categoricalCrossentropy(target: Tensor, output: Tensor, fromLogits?: boolean): Tensor;
/**
 * Categorical crossentropy with integer targets.
 *
 * @param target An integer tensor.
 * @param output A tensor resulting from a softmax (unless `fromLogits` is
 *  `true`, in which case `output` is expected to be the logits).
 * @param fromLogits Boolean, whether `output` is the result of a softmax, or is
 *   a tensor of logits.
 */
declare function sparseCategoricalCrossentropy(target: Tensor, output: Tensor, fromLogits?: boolean): Tensor;
/**
 * From TensorFlow's implementation in nn_impl.py:
 *
 * For brevity, let `x = logits`, `z = labels`.  The logistic loss is
 *      z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
 *    = z * -log(1 / (1 + exp(-x))) + (1 - z) * -log(exp(-x) / (1 + exp(-x)))
 *    = z * log(1 + exp(-x)) + (1 - z) * (-log(exp(-x)) + log(1 + exp(-x)))
 *    = z * log(1 + exp(-x)) + (1 - z) * (x + log(1 + exp(-x))
 *    = (1 - z) * x + log(1 + exp(-x))
 *    = x - x * z + log(1 + exp(-x))
 * For x < 0, to avoid overflow in exp(-x), we reformulate the above
 *      x - x * z + log(1 + exp(-x))
 *    = log(exp(x)) - x * z + log(1 + exp(-x))
 *    = - x * z + log(1 + exp(x))
 * Hence, to ensure stability and avoid overflow, the implementation uses this
 * equivalent formulation
 *    max(x, 0) - x * z + log(1 + exp(-abs(x)))
 *
 * @param labels The labels.
 * @param logits The logits.
 */
declare function sigmoidCrossEntropyWithLogits(labels: Tensor, logits: Tensor): Tensor;
declare function binaryCrossentropy(yTrue: Tensor, yPred: Tensor): Tensor;
declare function kullbackLeiblerDivergence(yTrue: Tensor, yPred: Tensor): Tensor;
declare function poisson(yTrue: Tensor, yPred: Tensor): Tensor;
declare function cosineProximity(yTrue: Tensor, yPred: Tensor): Tensor;
declare const mse: typeof meanSquaredError;
declare const MSE: typeof meanSquaredError;
declare const mae: typeof meanAbsoluteError;
declare const MAE: typeof meanAbsoluteError;
declare const mape: typeof meanAbsolutePercentageError;
declare const MAPE: typeof meanAbsolutePercentageError;
declare const msle: typeof meanSquaredLogarithmicError;
declare const MSLE: typeof meanSquaredLogarithmicError;
declare const kld: typeof kullbackLeiblerDivergence;
declare const KLD: typeof kullbackLeiblerDivergence;
declare const cosine: typeof cosineProximity;
declare const lossesMap: {
    [functionName: string]: LossOrMetricFn;
};
declare function get(identifierOrFn: string | LossOrMetricFn): LossOrMetricFn;

/// <amd-module name="@tensorflow/tfjs-layers/dist/keras_format/loss_config" />
/**
 * List of all known loss names.
 */
declare const lossOptions: ("mean_squared_error" | "mean_absolute_error" | "mean_absolute_percentage_error" | "mean_squared_logarithmic_error" | "squared_hinge" | "hinge" | "categorical_hinge" | "logcosh" | "categorical_crossentropy" | "sparse_categorical_crossentropy" | "kullback_leibler_divergence" | "poisson" | "cosine_proximity")[];
/**
 * A type representing the strings that are valid loss names.
 */
declare type LossIdentifier = typeof lossOptions[number];

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/loss_ops_utils" />
declare enum Reduction {
    NONE = 0,
    MEAN = 1,
    SUM = 2,
    SUM_BY_NONZERO_WEIGHTS = 3
}

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/lower_bound" />
/**
 * Searches for where a value would go in a sorted sequence.
 *
 * This is not a method for checking containment (like javascript in).
 *
 * The typical use case for this operation is "binning", "bucketing", or
 * "discretizing". The values are assigned to bucket-indices based on the edges
 * listed in 'sortedSequence'. This operation returns the bucket-index for each
 * value.
 *
 * The index returned corresponds to the first edge greater than or equal to the
 * value.
 *
 * The axis is not settable for this operation. It always operates on the
 * innermost dimension (axis=-1). The operation will accept any number of outer
 * dimensions.
 *
 * Note: This operation assumes that 'lowerBound' is sorted along the
 * innermost axis, maybe using 'sort(..., axis=-1)'. If the sequence is not
 * sorted no error is raised and the content of the returned tensor is not well
 * defined.
 *
 * ```js
 * const edges = tf.tensor1d([-1, 3.3, 9.1, 10.0]);
 * let values = tf.tensor1d([0.0, 4.1, 12.0]);
 * const result1 = tf.lowerBound(edges, values);
 * result1.print(); // [1, 2, 4]
 *
 * const seq = tf.tensor1d([0, 3, 9, 10, 10]);
 * values = tf.tensor1d([0, 4, 10]);
 * const result2 = tf.lowerBound(seq, values);
 * result2.print(); // [0, 2, 3]
 *
 * const sortedSequence = tf.tensor2d([[0., 3., 8., 9., 10.],
 *                                     [1., 2., 3., 4., 5.]]);
 * values = tf.tensor2d([[9.8, 2.1, 4.3],
 *                       [0.1, 6.6, 4.5, ]]);
 * const result3 = tf.lowerBound(sortedSequence, values);
 * result3.print(); // [[4, 1, 2], [0, 5, 4]]
 * ```
 * @param sortedSequence: N-D. Sorted sequence.
 * @param values: N-D. Search values.
 * @return An N-D int32 tensor the size of values containing the result of
 *     applying lower bound to each value. The result is not a global index to
 *     the entire Tensor, but the index in the last dimension.
 * @doc {heading: 'Operations', subheading: 'Evaluation'}
 */
declare function lowerBound(sortedSequence: Tensor | TensorLike, values: Tensor | TensorLike): Tensor;

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/lower_bound_test" />
{ };
/// <amd-module name="@tensorflow/tfjs-core/dist/gradients/LRN_grad" />
declare const lrnGradConfig: GradConfig;

/// <amd-module name="@tensorflow/tfjs-core/dist/math" />
/**
 * Exports under the tf.math.* namespace.
 */
{ confusionMatrix };

/// <amd-module name="@tensorflow/tfjs-layers/dist/utils/math_utils" />
declare type ArrayTypes = Uint8Array | Int32Array | Float32Array;
/**
 * Determine if a number is an integer.
 */
declare function isInteger(x: number): boolean;
/**
 * Calculate the product of an array of numbers.
 * @param array The array to calculate the product over.
 * @param begin Beginning index, inclusive.
 * @param end Ending index, exclusive.
 * @return The product.
 */
declare function arrayProd(array: number[] | ArrayTypes, begin?: number, end?: number): number;
/**
 * Compute minimum value.
 * @param array
 * @return minimum value.
 */
declare function min(array: number[] | Float32Array): number;
/**
 * Compute maximum value.
 * @param array
 * @return maximum value
 */
declare function max(array: number[] | Float32Array): number;
/**
 * Compute sum of array.
 * @param array
 * @return The sum.
 */
declare function sum(array: number[] | Float32Array): number;
/**
 * Compute mean of array.
 * @param array
 * @return The mean.
 */
declare function mean(array: number[] | Float32Array): number;
/**
 * Compute variance of array.
 * @param array
 * @return The variance.
 */
declare function variance(array: number[] | Float32Array): number;
/**
 * Compute median of array.
 * @param array
 * @return The median value.
 */
declare function median(array: number[] | Float32Array): number;
/**
 * Generate an array of integers in [begin, end).
 * @param begin Beginning integer, inclusive.
 * @param end Ending integer, exclusive.
 * @returns Range array.
 * @throws ValueError, iff `end` < `begin`.
 */
declare function range(begin: number, end: number): number[];
/// <amd-module name="@tensorflow/tfjs-core/dist/ops/mat_mul" />
/**
 * Computes the dot product of two matrices, A * B. These must be matrices.
 *
 * ```js
 * const a = tf.tensor2d([1, 2], [1, 2]);
 * const b = tf.tensor2d([1, 2, 3, 4], [2, 2]);
 *
 * a.matMul(b).print();  // or tf.matMul(a, b)
 * ```
 * @param a First matrix in dot product operation.
 * @param b Second matrix in dot product operation.
 * @param transposeA If true, `a` is transposed before multiplication.
 * @param transposeB If true, `b` is transposed before multiplication.
 *
 * @doc {heading: 'Operations', subheading: 'Matrices'}
 */
declare function matMul_<T extends Tensor>(a: Tensor | TensorLike, b: Tensor | TensorLike, transposeA?: boolean, transposeB?: boolean): T;
declare const matMul: typeof matMul_;
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/mat_mul_test" />
declare const MATMUL_SHARED_DIM_THRESHOLD = 1000;

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/max" />
/**
 * Computes the maximum of elements across dimensions of a `tf.Tensor`.
 *
 * Reduces the input along the dimensions given in `axes`. Unless `keepDims`
 * is true, the rank of the `tf.Tensor` is reduced by 1 for each entry in
 * `axes`. If `keepDims` is true, the reduced dimensions are retained with
 * length 1. If `axes` has no entries, all dimensions are reduced, and a
 * `tf.Tensor` with a single element is returned.
 *
 * ```js
 * const x = tf.tensor1d([1, 2, 3]);
 *
 * x.max().print();  // or tf.max(x)
 * ```
 *
 * ```js
 * const x = tf.tensor2d([1, 2, 3, 4], [2, 2]);
 *
 * const axis = 1;
 * x.max(axis).print();  // or tf.max(x, axis)
 * ```
 *
 * @param x The input tensor.
 * @param axis The dimension(s) to reduce. By default it reduces
 *     all dimensions.
 * @param keepDims If true, retains reduced dimensions with size 1.
 *
 * @doc {heading: 'Operations', subheading: 'Reduction'}
 */
declare function max_<T extends Tensor>(x: Tensor | TensorLike, axis?: number | number[], keepDims?: boolean): T;
declare const max: typeof max_;
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/maximum" />
/**
 * Returns the max of a and b (`a > b ? a : b`) element-wise.
 * Supports broadcasting.
 *
 * We also expose `tf.maximumStrict` which has the same signature as this op and
 * asserts that `a` and `b` are the same shape (does not broadcast).
 *
 * ```js
 * const a = tf.tensor1d([1, 4, 3, 16]);
 * const b = tf.tensor1d([1, 2, 9, 4]);
 *
 * a.maximum(b).print();  // or tf.maximum(a, b)
 * ```
 *
 * ```js
 * // Broadcast maximum a with b.
 * const a = tf.tensor1d([2, 4, 6, 8]);
 * const b = tf.scalar(5);
 *
 * a.maximum(b).print();  // or tf.maximum(a, b)
 * ```
 *
 * @param a The first tensor.
 * @param b The second tensor. Must have the same type as `a`.
 *
 * @doc {heading: 'Operations', subheading: 'Arithmetic'}
 */
declare function maximum_<T extends Tensor>(a: Tensor | TensorLike, b: Tensor | TensorLike): T;
declare const maximum: typeof maximum_;
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/gradients/Maximum_grad" />
declare const maximumGradConfig: GradConfig;

/// <amd-module name="@tensorflow/tfjs-core/dist/gradients/MaxPool3D_grad" />
declare const maxPool3DGradConfig: GradConfig;

/// <amd-module name="@tensorflow/tfjs-core/dist/gradients/MaxPool_grad" />
declare const maxPoolGradConfig: GradConfig;

/// <amd-module name="@tensorflow/tfjs-core/dist/gradients/Max_grad" />
declare const maxGradConfig: GradConfig;
/// <amd-module name="@tensorflow/tfjs-core/dist/public/chained_ops/max_pool" />

declare module '../../tensor' {
    interface Tensor<R extends Rank = Rank> {
        maxPool<T extends Tensor3D | Tensor4D>(filterSize: [number, number] | number, strides: [number, number] | number, pad: 'valid' | 'same' | number | ExplicitPadding, dimRoundingMode?: 'floor' | 'round' | 'ceil'): T;
    }
}

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/max_pool_3d" />
/**
 * Computes the 3D max pooling.
 *
 * ```js
 * const x = tf.tensor5d([1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 2, 2, 1]);
 * const result = tf.maxPool3d(x, 2, 1, 'valid');
 * result.print();
 * ```
 *
 * @param x The input tensor, of rank 5 or rank 4 of shape
 *     `[batch, depth, height, width, inChannels]`.
 * @param filterSize The filter size:
 *     `[filterDepth, filterHeight, filterWidth]`.
 *     If `filterSize` is a single number,
 *     then `filterDepth == filterHeight == filterWidth`.
 * @param strides The strides of the pooling:
 *     `[strideDepth, strideHeight, strideWidth]`.
 *     If `strides` is a single number,
 *     then `strideDepth == strideHeight == strideWidth`.
 * @param pad The type of padding algorithm.
 *    - `same` and stride 1: output will be of same size as input,
 *       regardless of filter size.
 *    - `valid`: output will be smaller than input if filter is larger
 *       than 1*1x1.
 *    - For more info, see this guide:
 *     [https://www.tensorflow.org/api_docs/python/tf/nn/convolution](
 *          https://www.tensorflow.org/api_docs/python/tf/nn/convolution)
 * @param dimRoundingMode A string from: 'ceil', 'round', 'floor'. If none is
 *     provided, it will default to truncate.
 * @param dataFormat An optional string from: "NDHWC", "NCDHW". Defaults to
 *     "NDHWC". Specify the data format of the input and output data. With the
 *     default format "NDHWC", the data is stored in the order of: [batch,
 *     depth, height, width, channels]. Only "NDHWC" is currently supported.
 * @doc {heading: 'Operations', subheading: 'Convolution'}
 */
declare function maxPool3d_<T extends Tensor4D | Tensor5D>(x: T | TensorLike, filterSize: [number, number, number] | number, strides: [number, number, number] | number, pad: 'valid' | 'same' | number, dimRoundingMode?: 'floor' | 'round' | 'ceil', dataFormat?: 'NDHWC' | 'NCDHW'): T;
declare const maxPool3d: typeof maxPool3d_;
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/max_pool_3d_grad" />
/**
 * Computes the backprop of a 3d max pool.
 *
 * @param dy The dy error, of rank 5 of shape
 *     [batchSize, depth, height, width, channels].
 * assumed.
 * @param input The original input image, of rank 5 or rank 4 of shape
 *     [batchSize, depth, height, width, channels].
 * @param output The original output image, of rank 5 of shape
 *     [batchSize, outDepth, outHeight, outWidth, channels].
 * @param filterSize The filter size:
 *     `[filterDepth, filterHeight, filterWidth]`.
 *     `filterSize` is a single number,
 *     then `filterDepth == filterHeight == filterWidth`.
 * @param strides The strides of the pooling:
 *     `[strideDepth, strideHeight, strideWidth]`. If
 *     `strides` is a single number, then `strideHeight == strideWidth`.
 * @param pad A string from: 'same', 'valid'. The type of padding algorithm
 *     used in the forward prop of the op.
 * @param dimRoundingMode A string from: 'ceil', 'round', 'floor'. If none is
 *     provided, it will default to truncate.
 */
declare function maxPool3dGrad_<T extends Tensor4D | Tensor5D>(dy: T | TensorLike, input: T | TensorLike, output: T | TensorLike, filterSize: [number, number, number] | number, strides: [number, number, number] | number, pad: 'valid' | 'same' | number, dimRoundingMode?: 'floor' | 'round' | 'ceil'): T;
declare const maxPool3dGrad: typeof maxPool3dGrad_;
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/max_pool_3d_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/max_pool_grad" />
/**
 * Computes the backprop of a 2D max pool.
 *
 * @param dy The dy error, of rank 4 or rank 3 of shape
 *     [batchSize, height, width, channels]. If rank 3, batch of 1 is
 * assumed.
 * @param input The original input image, of rank 4, of shape
 *     [batchSize, height, width, channels].
 * @param output The original output image, of rank 4, of shape
 *     [batchSize, outHeight, outWidth, channels].
 * @param filterSize The filter size: `[filterHeight, filterWidth]`. If
 *     `filterSize` is a single number, then `filterHeight == filterWidth`.
 * @param strides The strides of the pooling: `[strideHeight, strideWidth]`. If
 *     `strides` is a single number, then `strideHeight == strideWidth`.
 * @param pad The type of padding algorithm used in the forward prop of the op.
 *     'same', 'valid', for more info, see this guide:
 *     [https://www.tensorflow.org/api_docs/python/tf/nn/convolution](
 *          https://www.tensorflow.org/api_docs/python/tf/nn/convolution)
 * @param dimRoundingMode A string from: 'ceil', 'round', 'floor'. If none is
 *     provided, it will default to truncate.
 */
declare function maxPoolGrad_(dy: Tensor4D | TensorLike, input: Tensor4D | TensorLike, output: Tensor4D | TensorLike, filterSize: [number, number] | number, strides: [number, number] | number, pad: 'valid' | 'same' | number | conv_util.ExplicitPadding, dimRoundingMode?: 'floor' | 'round' | 'ceil'): Tensor4D;
declare const maxPoolGrad: typeof maxPoolGrad_;
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/max_pool_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/max_pool_with_argmax" />
/**
 * Computes the 2D max pooling of an image with Argmax index.
 * The indices in argmax are flattened, so that a maximum value at position `[b,
 * y, x, c]` becomes flattened index: `(y * width + x) * channels + c` if
 * include_batch_in_index is False; `((b * height + y) * width + x) * channels
 * +c` if include_batch_in_index is True.
 *
 * The indices returned are always in `[0, height) x [0, width)` before
 * flattening.
 *
 * @param x The input tensor, of rank 4 or rank 3 of shape
 *     `[batch, height, width, inChannels]`. If rank 3, batch of 1 is assumed.
 * @param filterSize The filter size: `[filterHeight, filterWidth]`. If
 *     `filterSize` is a single number, then `filterHeight == filterWidth`.
 * @param strides The strides of the pooling: `[strideHeight, strideWidth]`. If
 *     `strides` is a single number, then `strideHeight == strideWidth`.
 * @param dataFormat An optional string from: "NDHWC", "NCDHW". Defaults to
 *     "NDHWC". Specify the data format of the input and output data. With the
 *     default format "NDHWC", the data is stored in the order of: [batch,
 *     depth, height, width, channels]. Only "NDHWC" is currently supported.
 * @param pad The type of padding algorithm.
 *    - `same` and stride 1: output will be of same size as input,
 *       regardless of filter size.
 *    - `valid`: output will be smaller than input if filter is larger
 *       than 1x1.
 *    - For more info, see this guide:
 *     [https://www.tensorflow.org/api_docs/python/tf/nn/convolution](
 *          https://www.tensorflow.org/api_docs/python/tf/nn/convolution)
 * @param includeBatchIndex Defaults to False. Whether to include batch
 *    dimension in flattened index of argmax.
 *
 * @doc {heading: 'Operations', subheading: 'Convolution'}
 */
declare function maxPoolWithArgmax_<T extends Tensor4D>(x: T | TensorLike, filterSize: [number, number] | number, strides: [number, number] | number, pad: 'valid' | 'same' | number, includeBatchInIndex?: boolean): NamedTensorMap;
declare const maxPoolWithArgmax: typeof maxPoolWithArgmax_;
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/max_pool_with_argmax_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/max_test" />
{ };
/// <amd-module name="@tensorflow/tfjs-core/dist/public/chained_ops/mean" />
declare module '../../tensor' {
    interface Tensor<R extends Rank = Rank> {
        mean<T extends Tensor>(axis?: number | number[], keepDims?: boolean): T;
    }
}

/// <amd-module name="@tensorflow/tfjs-core/dist/gradients/Mean_grad" />
declare const meanGradConfig: GradConfig;

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/losses/mean_squared_error" />
/**
 * Computes the mean squared error between two tensors.
 *
 * @param labels The ground truth output tensor, same dimensions as
 *    'predictions'.
 * @param predictions The predicted outputs.
 * @param weights Tensor whose rank is either 0, or the same rank as
 *    `labels`, and must be broadcastable to `labels` (i.e., all dimensions
 *    must be either `1`, or the same as the corresponding `losses`
 *    dimension).
 * @param reduction Type of reduction to apply to loss. Should be of type
 *    `Reduction`
 *
 * @doc {heading: 'Training', subheading: 'Losses', namespace: 'losses'}
 */
declare function meanSquaredError_<T extends Tensor, O extends Tensor>(labels: T | TensorLike, predictions: T | TensorLike, weights?: Tensor | TensorLike, reduction?: Reduction): O;
declare const meanSquaredError: typeof meanSquaredError_;
{ };
/// <amd-module name="@tensorflow/tfjs-core/dist/ops/losses/mean_squared_error_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/mean_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-layers/dist/layers/merge" />
/**
 * Generic Merge layer for element-wise merge functions.
 *
 * Used to implement `Sum`, `Average`, `Concatenate`, etc.
 */
declare abstract class Merge extends Layer {
    protected reshapeRequired: boolean;
    constructor(args?: LayerArgs);
    /**
     * Logic for merging multiple tensors, to be overridden by subclasses.
     * @param inputs
     */
    protected mergeFunction(inputs: Tensor[]): Tensor;
    /**
     * Computes the shape of the result of an elementwise operation.
     *
     * @param shape1: Shape of the first tensor.
     * @param shape2: Shape of the second tensor.
     * @returns Expected output shape when an elementwise operation is carried
     *   out on 2 tensors with shapes `shape1` and `shape2`.
     * @throws ValueError: If `shape1` and `shape2` are not compatible for
     *   element-wise operations.
     */
    private computeElementwiseOpOutputShape;
    build(inputShape: Shape | Shape[]): void;
    call(inputs: Tensor | Tensor[], kwargs: Kwargs): Tensor | Tensor[];
    computeOutputShape(inputShape: Shape | Shape[]): Shape | Shape[];
    computeMask(inputs: Tensor | Tensor[], mask?: Tensor | Tensor[]): Tensor;
}
declare class Add extends Merge {
    /** @nocollapse */
    static className: string;
    constructor(args?: LayerArgs);
    protected mergeFunction(inputs: Tensor[]): Tensor;
}
/**
 * Calculate the element-wise sum of inputs, which all have the same shape.
 *
 * This function can be invoked in three ways.
 *
 * 1. Construct an instance of `Add` layer, by using no input argument
 *    or a single configuration argument. The resultant `Add` layer can then
 *    be used on `tf.SymbolicTensor`s or `tf.Tensor`s. For example:
 *
 * ```js
 * const addLayer = tf.layers.add();
 *
 * // The layer can be applied to inputs.
 * const input1 = tf.input({shape: [2, 2]});
 * const input2 = tf.input({shape: [2, 2]});
 * const output = addLayer.apply([input1, input2]);
 * console.log(output.shape);
 * // You get [null, 2, 2], with the first dimension as the undetermined batch
 * // dimension.
 * ```
 *
 * 2. Invoke directly on an `Array` of `tf.SymbolicTensor`s. This constructs
 *    an `Layer` object internally and calls its `apply` method on the inputs,
 *    generating a new `tf.SymbolicTensor`. For example:
 *
 * ```js
 * const input1 = tf.input({shape: [2, 2]});
 * const input2 = tf.input({shape: [2, 2]});
 * const output = tf.layers.add([input1, input2]);
 * console.log(output.shape);
 * // You get [null, 2, 2], with the first dimension as the undetermined batch
 * // dimension.
 * ```
 *
 * 3. Invoke directly on `tf.Tensor`s, i.e., concrete values. This constructs
 *    an `Layer` object internally and calls its `apply` method on the inputs,
 *    generating a new `tf.Tensor` as the result of the computation. For
 * example:
 *
 * ```js
 * const input1 = tf.tensor2d([1, 2, 3, 4], [2, 2]);
 * const input2 = tf.tensor2d([10, 20, 30, 40], [2, 2]);
 * tf.layers.add([input1, input2]).print();
 * // Gives [[11, 22], [33, 44]].
 *
 */
declare function add(config?: SymbolicTensor[] | Tensor[] | LayerArgs): Layer | SymbolicTensor | Tensor;
declare class Multiply extends Merge {
    /** @nocollapse */
    static className: string;
    constructor(args?: LayerArgs);
    protected mergeFunction(inputs: Tensor[]): Tensor;
}
/**
 * Calculate the element-wise product of inputs, which all have the same shape.
 *
 * This function can be invoked in three ways.
 *
 * 1. Construct an instance of `Multiply` layer, by using no input argument
 *    or a single configuration argument. The resultant `Multiply` layer can
 *    then be used on `tf.SymbolicTensor`s or `tf.Tensor`s. For example:
 *
 * ```js
 * const multiplyLayer = tf.layers.multiply();
 *
 * // The layer can be applied to inputs.
 * const input1 = tf.input({shape: [2, 2]});
 * const input2 = tf.input({shape: [2, 2]});
 * const output = multiplyLayer.apply([input1, input2]);
 * console.log(output.shape);
 * // You get [null, 2, 2], with the first dimension as the undetermined batch
 * // dimension.
 * ```
 *
 * 2. Invoke directly on an `Array` of `tf.SymbolicTensor`s. This constructs
 *    an `Layer` object internally and calls its `apply` method on the inputs,
 *    generating a new `tf.SymbolicTensor`. For example:
 *
 * ```js
 * const input1 = tf.input({shape: [2, 2]});
 * const input2 = tf.input({shape: [2, 2]});
 * const output = tf.layers.multiply([input1, input2]);
 * console.log(output.shape);
 * // You get [null, 2, 2], with the first dimension as the undetermined batch
 * // dimension.
 * ```
 *
 * 3. Invoke directly on `tf.Tensor`s, i.e., concrete values. This constructs
 *    an `Layer` object internally and calls its `apply` method on the inputs,
 *    generating a new `tf.Tensor` as the result of the computation. For
 * example:
 *
 * ```js
 * const input1 = tf.tensor2d([1, 2, 3, 4], [2, 2]);
 * const input2 = tf.tensor2d([10, 20, 30, 40], [2, 2]);
 * tf.layers.multiply([input1, input2]).print();
 * // Gives [[10, 40], [90, 160]].
 *
 */
declare function multiply(config?: SymbolicTensor[] | Tensor[] | LayerArgs): Layer | SymbolicTensor | Tensor;
declare class Average extends Merge {
    /** @nocollapse */
    static className: string;
    constructor(args?: LayerArgs);
    protected mergeFunction(inputs: Tensor[]): Tensor;
}
/**
 * Calculate the element-wise arithmetic mean of inputs, which all have the same
 * shape.
 *
 * This function can be invoked in three ways.
 *
 * 1. Construct an instance of `Average` layer, by using no input argument
 *    or a single configuration argument. The resultant `Average` layer can then
 *    be used on `tf.SymbolicTensor`s or `tf.Tensor`s. For example:
 *
 * ```js
 * const averageLayer = tf.layers.average();
 *
 * // The layer can be applied to inputs.
 * const input1 = tf.input({shape: [2, 2]});
 * const input2 = tf.input({shape: [2, 2]});
 * const output = averageLayer.apply([input1, input2]);
 * console.log(output.shape);
 * // You get [null, 2, 2], with the first dimension as the undetermined batch
 * // dimension.
 * ```
 *
 * 2. Invoke directly on an `Array` of `tf.SymbolicTensor`s. This constructs
 *    an `Layer` object internally and calls its `apply` method on the inputs,
 *    generating a new `tf.SymbolicTensor`. For example:
 *
 * ```js
 * const input1 = tf.input({shape: [2, 2]});
 * const input2 = tf.input({shape: [2, 2]});
 * const output = tf.layers.average([input1, input2]);
 * console.log(output.shape);
 * // You get [null, 2, 2], with the first dimension as the undetermined batch
 * // dimension.
 * ```
 *
 * 3. Invoke directly on `tf.Tensor`s, i.e., concrete values. This constructs
 *    an `Layer` object internally and calls its `apply` method on the inputs,
 *    generating a new `tf.Tensor` as the result of the computation. For
 * example:
 *
 * ```js
 * const input1 = tf.tensor2d([1, 2, 3, 4], [2, 2]);
 * const input2 = tf.tensor2d([10, 20, 30, 40], [2, 2]);
 * tf.layers.average([input1, input2]).print();
 * // Gives [[5.5, 11], [16.5, 22]].
 *
 */
declare function average(config?: SymbolicTensor[] | Tensor[] | LayerArgs): Layer | SymbolicTensor | Tensor;
declare class Maximum extends Merge {
    /** @nocollapse */
    static className: string;
    constructor(args?: LayerArgs);
    protected mergeFunction(inputs: Tensor[]): Tensor;
}
/**
 * Calculate the element-wise maximum of inputs, which all have the same shape.
 *
 * This function can be invoked in three ways.
 *
 * 1. Construct an instance of `Maximum` layer, by using no input argument
 *    or a single configuration argument. The resultant `Maximum` layer can then
 *    be used on `tf.SymbolicTensor`s or `tf.Tensor`s. For example:
 *
 * ```js
 * const maximumLayer = tf.layers.maximum();
 *
 * // The layer can be applied to inputs.
 * const input1 = tf.input({shape: [2, 2]});
 * const input2 = tf.input({shape: [2, 2]});
 * const output = maximumLayer.apply([input1, input2]);
 * console.log(output.shape);
 * // You get [null, 2, 2], with the first dimension as the undetermined batch
 * // dimension.
 * ```
 *
 * 2. Invoke directly on an `Array` of `tf.SymbolicTensor`s. This constructs
 *    an `Layer` object internally and calls its `apply` method on the inputs,
 *    generating a new `tf.SymbolicTensor`. For example:
 *
 * ```js
 * const input1 = tf.input({shape: [2, 2]});
 * const input2 = tf.input({shape: [2, 2]});
 * const output = tf.layers.maximum([input1, input2]);
 * console.log(output.shape);
 * // You get [null, 2, 2], with the first dimension as the undetermined batch
 * // dimension.
 * ```
 *
 * 3. Invoke directly on `tf.Tensor`s, i.e., concrete values. This constructs
 *    an `Layer` object internally and calls its `apply` method on the inputs,
 *    generating a new `tf.Tensor` as the result of the computation. For
 * example:
 *
 * ```js
 * const input1 = tf.tensor2d([1, 20, 3, 40], [2, 2]);
 * const input2 = tf.tensor2d([10, 2, 30, 4], [2, 2]);
 * tf.layers.maximum([input1, input2]).print();
 * // Gives [[10, 20], [30, 40]].
 *
 */
declare function maximum(config?: SymbolicTensor[] | Tensor[] | LayerArgs): Layer | SymbolicTensor | Tensor;
declare class Minimum extends Merge {
    /** @nocollapse */
    static className: string;
    constructor(args?: LayerArgs);
    protected mergeFunction(inputs: Tensor[]): Tensor;
}
/**
 * Calculate the element-wise minimum of inputs, which all have the same shape.
 *
 * This function can be invoked in three ways.
 *
 * 1. Construct an instance of `Minimum` layer, by using no input argument
 *    or a single configuration argument. The resultant `Minimum` layer can then
 *    be used on `tf.SymbolicTensor`s or `tf.Tensor`s. For example:
 *
 * ```js
 * const minimumLayer = tf.layers.minimum();
 *
 * // The layer can be applied to inputs.
 * const input1 = tf.input({shape: [2, 2]});
 * const input2 = tf.input({shape: [2, 2]});
 * const output = minimumLayer.apply([input1, input2]);
 * console.log(output.shape);
 * // You get [null, 2, 2], with the first dimension as the undetermined batch
 * // dimension.
 * ```
 *
 * 2. Invoke directly on an `Array` of `tf.SymbolicTensor`s. This constructs
 *    an `Layer` object internally and calls its `apply` method on the inputs,
 *    generating a new `tf.SymbolicTensor`. For example:
 *
 * ```js
 * const input1 = tf.input({shape: [2, 2]});
 * const input2 = tf.input({shape: [2, 2]});
 * const output = tf.layers.minimum([input1, input2]);
 * console.log(output.shape);
 * // You get [null, 2, 2], with the first dimension as the undetermined batch
 * // dimension.
 * ```
 *
 * 3. Invoke directly on `tf.Tensor`s, i.e., concrete values. This constructs
 *    an `Layer` object internally and calls its `apply` method on the inputs,
 *    generating a new `tf.Tensor` as the result of the computation. For
 * example:
 *
 * ```js
 * const input1 = tf.tensor2d([1, 20, 3, 40], [2, 2]);
 * const input2 = tf.tensor2d([10, 2, 30, 4], [2, 2]);
 * tf.layers.minimum([input1, input2]).print();
 * // Gives [[1, 2], [3, 4]].
 *
 */
declare function minimum(config?: SymbolicTensor[] | Tensor[] | LayerArgs): Layer | SymbolicTensor | Tensor;
declare interface ConcatenateLayerArgs extends LayerArgs {
    /**
     * Axis along which to concatenate.
     */
    axis?: number;
}
declare class Concatenate extends Merge {
    /** @nocollapse */
    static className: string;
    readonly DEFAULT_AXIS = -1;
    private readonly axis;
    constructor(args?: ConcatenateLayerArgs);
    build(inputShape: Shape | Shape[]): void;
    protected mergeFunction(inputs: Tensor[]): Tensor;
    computeOutputShape(inputShape: Shape | Shape[]): Shape | Shape[];
    computeMask(inputs: Tensor | Tensor[], mask?: Tensor | Tensor[]): Tensor;
    getConfig(): serialization.ConfigDict;
}
/**
 * Concatenate an `Array` of inputs.
 *
 * This function can be invoked in three ways.
 *
 * 1. Construct an instance of `Concatenate` layer, by using no input argument
 *    or a single configuration argument. The resultant `Concatenate` layer can
 *    then be used on `tf.SymbolicTensor`s or `tf.Tensor`s. For example:
 *
 * ```js
 * const concatLayer = tf.layers.concatenate();
 *
 * // The layer can be applied to inputs.
 * const input1 = tf.input({shape: [2, 3]});
 * const input2 = tf.input({shape: [2, 4]});
 * const output = concatLayer.apply([input1, input2]);
 * console.log(output.shape);
 * // You get [null, 2, 7], with the first dimension as the undetermined batch
 * // dimension and the last dimension as the result of concatenating the
 * // last dimensions of the two inputs.
 * ```
 *
 * 2. Invoke directly on an `Array` of `tf.SymbolicTensor`s. This constructs
 *    an `Layer` object internally and calls its `apply` method on the inputs,
 *    generating a new `tf.SymbolicTensor`. For example:
 *
 * ```js
 * const input1 = tf.input({shape: [2, 3]});
 * const input2 = tf.input({shape: [2, 4]});
 * const output = tf.layers.concatenate([input1, input2]);
 * console.log(output.shape);
 * // You get [null, 2, 2], with the first dimension as the undetermined batch
 * // dimension and the last dimension as the result of concatenating the
 * // last dimensions of the two inputs.
 * ```
 *
 * 3. Invoke directly on `tf.Tensor`s, i.e., concrete values. This constructs
 *    an `Layer` object internally and calls its `apply` method on the inputs,
 *    generating a new `tf.Tensor` as the result of the computation. For
 * example:
 *
 * ```js
 * const input1 = tf.tensor2d([[1, 2], [3, 4]], [2, 2]);
 * const input2 = tf.tensor2d([[10, 20], [30, 40]], [2, 2]);
 * tf.layers.concatenate([input1, input2]).print();
 * // Gives [[1, 2, 10, 20], [3, 4, 30, 40]].
 *
 */
declare function concatenate(config?: SymbolicTensor[] | Tensor[] | ConcatenateLayerArgs): Layer | SymbolicTensor | Tensor;
declare interface DotLayerArgs extends LayerArgs {
    /**
     * Axis or axes along which the dot product will be taken.
     *
     * Integer or an Array of integers.
     */
    axes: number | [number, number];
    /**
     * Whether to L2-normalize samples along the dot product axis
     * before taking the dot product.
     *
     * If set to `true`, the output of the dot product is the cosine
     * proximity between the two samples.
     */
    normalize?: boolean;
}
declare class Dot extends Merge {
    /** @nocollapse */
    static className: string;
    private axes;
    private normalize;
    constructor(args: DotLayerArgs);
    build(inputShape: Shape | Shape[]): void;
    protected mergeFunction(inputs: Tensor[]): Tensor;
    private interpretAxes;
    computeOutputShape(inputShape: Shape | Shape[]): Shape | Shape[];
    computeMask(inputs: Tensor | Tensor[], mask?: Tensor | Tensor[]): Tensor;
    getConfig(): serialization.ConfigDict;
}

/// <amd-module name="@tensorflow/tfjs-layers/dist/keras_format/layers/merge_serialization" />
declare type AddLayerSerialization = BaseLayerSerialization<'Add', LayerConfig>;
declare type MultiplyLayerSerialization = BaseLayerSerialization<'Multiply', LayerConfig>;
declare type AverageLayerSerialization = BaseLayerSerialization<'Average', LayerConfig>;
declare type MaximumLayerSerialization = BaseLayerSerialization<'Maximum', LayerConfig>;
declare type MinimumLayerSerialization = BaseLayerSerialization<'Minimum', LayerConfig>;
interface ConcatenateLayerConfig extends LayerConfig {
    axis?: number;
}
declare type ConcatenateLayerSerialization = BaseLayerSerialization<'Concatenate', ConcatenateLayerConfig>;
interface DotLayerConfig extends LayerConfig {
    axes: number | [number, number];
    normalize?: boolean;
}
declare type DotLayerSerialization = BaseLayerSerialization<'Dot', DotLayerConfig>;
declare type MergeLayerSerialization = AddLayerSerialization | MultiplyLayerSerialization | AverageLayerSerialization | MaximumLayerSerialization | MinimumLayerSerialization | ConcatenateLayerSerialization | DotLayerSerialization;
declare type MergeLayerClassName = MergeLayerSerialization['class_name'];
/**
 * A string array of valid MergeLayer class names.
 *
 * This is guaranteed to match the `MergeLayerClassName` union type.
 */
declare const mergeLayerClassNames: MergeLayerClassName[];

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/meshgrid" />
/**
 * Broadcasts parameters for evaluation on an N-D grid.
 *
 * Given N one-dimensional coordinate arrays `*args`, returns a list `outputs`
 * of N-D coordinate arrays for evaluating expressions on an N-D grid.
 *
 * Notes:
 * `meshgrid` supports cartesian ('xy') and matrix ('ij') indexing conventions.
 * When the `indexing` argument is set to 'xy' (the default), the broadcasting
 * instructions for the first two dimensions are swapped.
 * Examples:
 * Calling `const [X, Y] = meshgrid(x, y)` with the tensors
 *
 * ```javascript
 * const x = [1, 2, 3];
 * const y = [4, 5, 6];
 * const [X, Y] = tf.meshgrid(x, y);
 * // X = [[1, 2, 3],
 * //      [1, 2, 3],
 * //      [1, 2, 3]]
 * // Y = [[4, 4, 4],
 * //      [5, 5, 5],
 * //      [6, 6, 6]]
 * ```
 *
 * @param x Tensor with rank geq 1.
 * @param y Tensor with rank geq 1.
 * @param indexing
 *
 * @doc {heading: 'Operations', subheading: 'Slicing and Joining'}
 */
declare function meshgrid<T extends Tensor>(x?: T | TensorLike, y?: T | TensorLike, { indexing }?: {
    indexing?: string;
}): T[];

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/meshgrid_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-layers/dist/metrics" />
declare function binaryAccuracy(yTrue: Tensor, yPred: Tensor): Tensor;
declare function categoricalAccuracy(yTrue: Tensor, yPred: Tensor): Tensor;
declare function precision(yTrue: Tensor, yPred: Tensor): Tensor;
declare function recall(yTrue: Tensor, yPred: Tensor): Tensor;
declare function binaryCrossentropy(yTrue: Tensor, yPred: Tensor): Tensor;
declare function sparseCategoricalAccuracy(yTrue: Tensor, yPred: Tensor): Tensor;
declare function topKCategoricalAccuracy(yTrue: Tensor, yPred: Tensor): Tensor;
declare function sparseTopKCategoricalAccuracy(yTrue: Tensor, yPred: Tensor): Tensor;
declare const mse: typeof meanSquaredError;
declare const MSE: typeof meanSquaredError;
declare const mae: typeof meanAbsoluteError;
declare const MAE: typeof meanAbsoluteError;
declare const mape: typeof meanAbsolutePercentageError;
declare const MAPE: typeof meanAbsolutePercentageError;
declare const categoricalCrossentropy: typeof categoricalCrossentropyLoss;
declare const cosine: typeof cosineProximity;
declare const sparseCategoricalCrossentropy: typeof sparseCategoricalCrossentropyLoss;
declare const metricsMap: {
    [functionName: string]: LossOrMetricFn;
};
declare function get(identifier: string | LossOrMetricFn): LossOrMetricFn;
/**
 * Get the shortcut function name.
 *
 * If the fn name is a string,
 *   directly return the string name.
 * If the function is included in metricsMap or lossesMap,
 *   return key of the map.
 *   - If the function relative to multiple keys,
 *     return the first found key as the function name.
 *   - If the function exists in both lossesMap and metricsMap,
 *     search lossesMap first.
 * If the function is not included in metricsMap or lossesMap,
 *   return the function name.
 *
 * @param fn loss function, metric function, or short cut name.
 * @returns Loss or Metric name in string.
 */
declare function getLossOrMetricName(fn: string | LossOrMetricFn): string;

/// <amd-module name="@tensorflow/tfjs-data/dist/iterators/microphone_iterator" />
/**
 * Provide a stream of tensors from microphone audio stream. The tensors are
 * representing audio data as frequency-domain spectrogram generated with
 * browser's native FFT. Tensors representing time-domain waveform is available
 * based on configuration. Only works in browser environment.
 */
declare class MicrophoneIterator extends LazyIterator<TensorContainer> {
    protected readonly microphoneConfig: MicrophoneConfig;
    private isClosed;
    private stream;
    private readonly fftSize;
    private readonly columnTruncateLength;
    private freqData;
    private timeData;
    private readonly numFrames;
    private analyser;
    private audioContext;
    private sampleRateHz;
    private readonly audioTrackConstraints;
    private readonly smoothingTimeConstant;
    private readonly includeSpectrogram;
    private readonly includeWaveform;
    private constructor();
    summary(): string;
    static create(microphoneConfig?: MicrophoneConfig): Promise<MicrophoneIterator>;
    start(): Promise<void>;
    next(): Promise<IteratorResult<TensorContainer>>;
    capture(): Promise<{
        spectrogram: Tensor3D;
        waveform: Tensor2D;
    }>;
    private getAudioData;
    stop(): void;
    toArray(): Promise<Tensor[]>;
    getSampleRate(): number;
    private flattenQueue;
    private getTensorFromAudioDataArray;
}
/// <amd-module name="@tensorflow/tfjs-core/dist/ops/min" />
/**
 * Computes the minimum value from the input.
 *
 * Reduces the input along the dimensions given in `axes`. Unless `keepDims`
 * is true, the rank of the array is reduced by 1 for each entry in `axes`.
 * If `keepDims` is true, the reduced dimensions are retained with length 1.
 * If `axes` has no entries, all dimensions are reduced, and an array with a
 * single element is returned.
 *
 * ```js
 * const x = tf.tensor1d([1, 2, 3]);
 *
 * x.min().print();  // or tf.min(x)
 * ```
 *
 * ```js
 * const x = tf.tensor2d([1, 2, 3, 4], [2, 2]);
 *
 * const axis = 1;
 * x.min(axis).print();  // or tf.min(x, axis)
 * ```
 *
 * @param x The input Tensor.
 * @param axis The dimension(s) to reduce. By default it reduces
 *     all dimensions.
 * @param keepDims If true, retains reduced dimensions with size 1.
 *
 * @doc {heading: 'Operations', subheading: 'Reduction'}
 */
declare function min_<T extends Tensor>(x: Tensor | TensorLike, axis?: number | number[], keepDims?: boolean): T;
declare const min: typeof min_;
{ };
/// <amd-module name="@tensorflow/tfjs-core/dist/public/chained_ops/minimum" />
declare module '../../tensor' {
    interface Tensor<R extends Rank = Rank> {
        minimum<T extends Tensor>(b: Tensor | TensorLike): T;
    }
}

/// <amd-module name="@tensorflow/tfjs-core/dist/gradients/Minimum_grad" />
declare const minimumGradConfig: GradConfig;

/// <amd-module name="@tensorflow/tfjs-core/dist/gradients/Min_grad" />
declare const minGradConfig: GradConfig;

/// <amd-module name="@tensorflow/tfjs-core/dist/gradients/min_max_grad_util" />
/**
 * Gradient helper function for the min and max operations.
 */
declare function gradForMinAndMax<T extends Tensor>(dy: T, y: T, xOrig: Tensor, origAxes: number[]): {
    x: () => Tensor<import("@tensorflow/tfjs-core/dist/types").Rank>;
};

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/min_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/gradients/MirrorPad_grad" />
declare const mirrorPadGradConfig: GradConfig;

/// <amd-module name="@tensorflow/tfjs-core/dist/public/chained_ops/mirror_pad" />
declare module '../../tensor' {
    interface Tensor<R extends Rank = Rank> {
        mirrorPad<T extends Tensor>(paddings: Array<[number, number]>, mode: 'reflect' | 'symmetric'): T;
    }
}

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/mirror_pad_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/mod" />
/**
 * Returns the mod of a and b element-wise.
 * `floor(x / y) * y + mod(x, y) = x`
 * Supports broadcasting.
 *
 * We also expose `tf.modStrict` which has the same signature as this op and
 * asserts that `a` and `b` are the same shape (does not broadcast).
 *
 * ```js
 * const a = tf.tensor1d([1, 4, 3, 16]);
 * const b = tf.tensor1d([1, 2, 9, 4]);
 *
 * a.mod(b).print();  // or tf.mod(a, b)
 * ```
 *
 * ```js
 * // Broadcast a mod b.
 * const a = tf.tensor1d([2, 4, 6, 8]);
 * const b = tf.scalar(5);
 *
 * a.mod(b).print();  // or tf.mod(a, b)
 * ```
 *
 * @param a The first tensor.
 * @param b The second tensor. Must have the same type as `a`.
 *
 * @doc {heading: 'Operations', subheading: 'Arithmetic'}
 */
declare function mod_<T extends Tensor>(a: Tensor | TensorLike, b: Tensor | TensorLike): T;
declare const mod: typeof mod_;
{ };

/// <amd-module name="@tensorflow/tfjs-layers/dist/models" />
/**
 * Parses a JSON model configuration file and returns a model instance.
 *
 * ```js
 * // This example shows how to serialize a model using `toJSON()` and
 * // deserialize it as another model using `tf.models.modelFromJSON()`.
 * // Note: this example serializes and deserializes only the topology
 * // of the model; the weights of the loaded model will be different
 * // from those of the the original model, due to random weight
 * // initialization.
 * // To load the topology and weights of a model, use `tf.loadLayersModel()`.
 * const model1 = tf.sequential();
 * model1.add(tf.layers.repeatVector({inputShape: [2], n: 4}));
 * // Serialize `model1` as a JSON object.
 * const model1JSON = model1.toJSON(null, false);
 * model1.summary();
 *
 * const model2 = await tf.models.modelFromJSON(model1JSON);
 * model2.summary();
 * ```
 *
 *  @param modelAndWeightsConfig JSON object or string encoding a model and
 *       weights configuration. It can also be only the topology JSON of the
 *       model, in which case the weights will not be loaded.
 *  @param custom_objects Optional dictionary mapping names
 *       (strings) to custom classes or functions to be
 *       considered during deserialization.
 * @returns A TensorFlow.js Layers `tf.LayersModel` instance (uncompiled).
 */
declare function modelFromJSON(modelAndWeightsConfig: ModelAndWeightsConfig | PyJsonDict, customObjects?: serialization.ConfigDict): Promise<LayersModel>;
/**
 * Options for loading a saved mode in TensorFlow.js format.
 */
interface ModelAndWeightsConfig {
    /**
     * A JSON object or JSON string containing the model config.
     *
     * This can be either of the following two formats:
     *   - A model archiecture-only config,  i.e., a format consistent with the
     *     return value of`keras.Model.to_json()`.
     *   - A full model config, containing not only model architecture, but also
     *     training options and state, i.e., a format consistent with the return
     *     value of `keras.models.save_model()`.
     */
    modelTopology: PyJsonDict;
    /**
     * A weights manifest in TensorFlow.js format.
     */
    weightsManifest?: io.WeightsManifestConfig;
    /**
     * Path to prepend to the paths in `weightManifest` before fetching.
     *
     * The path may optionally end in a slash ('/').
     */
    pathPrefix?: string;
}
interface ModelPredictArgs {
    /**
     * Optional. Batch size (Integer). If unspecified, it will default to 32.
     */
    batchSize?: number;
    /**
     * Optional. Verbosity mode. Defaults to false.
     */
    verbose?: boolean;
}
/**
 * Load a model composed of Layer objects, including its topology and optionally
 * weights. See the Tutorial named "How to import a Keras Model" for usage
 * examples.
 *
 * This method is applicable to:
 *
 * 1. Models created with the `tf.layers.*`, `tf.sequential`, and
 * `tf.model` APIs of TensorFlow.js and later saved with the
 * `tf.LayersModel.save` method.
 * 2. Models converted from Keras or TensorFlow tf.keras using the
 * [tensorflowjs_converter](https://github.com/tensorflow/tfjs/tree/master/tfjs-converter).
 *
 * This mode is *not* applicable to TensorFlow `SavedModel`s or their converted
 * forms. For those models, use `tf.loadGraphModel`.
 *
 * Example 1. Load a model from an HTTP server.
 *
 * ```js
 * const model = await tf.loadLayersModel(
 *     'https://storage.googleapis.com/tfjs-models/tfjs/iris_v1/model.json');
 * model.summary();
 * ```
 *
 * Example 2: Save `model`'s topology and weights to browser [local
 * storage](https://developer.mozilla.org/en-US/docs/Web/API/Window/localStorage);
 * then load it back.
 *
 * ```js
 * const model = tf.sequential(
 *     {layers: [tf.layers.dense({units: 1, inputShape: [3]})]});
 * console.log('Prediction from original model:');
 * model.predict(tf.ones([1, 3])).print();
 *
 * const saveResults = await model.save('localstorage://my-model-1');
 *
 * const loadedModel = await tf.loadLayersModel('localstorage://my-model-1');
 * console.log('Prediction from loaded model:');
 * loadedModel.predict(tf.ones([1, 3])).print();
 * ```
 *
 * Example 3. Saving `model`'s topology and weights to browser
 * [IndexedDB](https://developer.mozilla.org/en-US/docs/Web/API/IndexedDB_API);
 * then load it back.
 *
 * ```js
 * const model = tf.sequential(
 *     {layers: [tf.layers.dense({units: 1, inputShape: [3]})]});
 * console.log('Prediction from original model:');
 * model.predict(tf.ones([1, 3])).print();
 *
 * const saveResults = await model.save('indexeddb://my-model-1');
 *
 * const loadedModel = await tf.loadLayersModel('indexeddb://my-model-1');
 * console.log('Prediction from loaded model:');
 * loadedModel.predict(tf.ones([1, 3])).print();
 * ```
 *
 * Example 4. Load a model from user-selected files from HTML
 * [file input
 * elements](https://developer.mozilla.org/en-US/docs/Web/HTML/Element/input/file).
 *
 * ```js
 * // Note: this code snippet will not work without the HTML elements in the
 * //   page
 * const jsonUpload = document.getElementById('json-upload');
 * const weightsUpload = document.getElementById('weights-upload');
 *
 * const model = await tf.loadLayersModel(
 *     tf.io.browserFiles([jsonUpload.files[0], weightsUpload.files[0]]));
 * ```
 *
 * @param pathOrIOHandler Can be either of the two formats
 *   1. A string path to the `ModelAndWeightsConfig` JSON describing
 *      the model in the canonical TensorFlow.js format. For file://
 *      (tfjs-node-only), http:// and https:// schemas, the path can be
 *      either absolute or relative. The content of the JSON file is assumed to
 *      be a JSON object with the following fields and values:
 *      - 'modelTopology': A JSON object that can be either of:
 *        1. a model architecture JSON consistent with the format of the return
 *            value of `keras.Model.to_json()`
 *        2. a full model JSON in the format of `keras.models.save_model()`.
 *      - 'weightsManifest': A TensorFlow.js weights manifest.
 *      See the Python converter function `save_model()` for more details.
 *      It is also assumed that model weights can be accessed from relative
 *      paths described by the `paths` fields in weights manifest.
 *   2. A `tf.io.IOHandler` object that loads model artifacts with its `load`
 *      method.
 * @param options Optional configuration arguments for the model loading,
 *   including:
 *   - `strict`: Require that the provided weights exactly match those required
 *     by the layers.  Default true.  Passing false means that both extra
 *     weights and missing weights will be silently ignored.
 *   - `onProgress`: A progress callback of the form:
 *     `(fraction: number) => void`. This callback can be used to monitor the
 *     model-loading process.
 * @returns A `Promise` of `tf.LayersModel`, with the topology and weights
 *     loaded.
 *
 * @doc {heading: 'Models', subheading: 'Loading'}
 */
declare function loadLayersModel(pathOrIOHandler: string | io.IOHandler, options?: io.LoadOptions): Promise<LayersModel>;
/**
 * Load a model and optionally its weights, using an IOHandler object.
 *
 * @param handler The instance of `IOHandler` to be used during the model
 *   loading.
 * @param customObjects Any optional custom objects to be used during model
 *   loading.
 * @param strict Whether the weight loading will be done in strict mode.
 *   Default: `true`.
 */
declare function loadLayersModelFromIOHandler(handler: io.IOHandler, customObjects?: serialization.ConfigDict, options?: io.LoadOptions): Promise<LayersModel>;
/**
 * Configuration for a Sequential model.
 */
interface SequentialArgs {
    /** Stack of layers for the model. */
    layers?: Layer[];
    /** The name of this model. */
    name?: string;
}
/**
 * A model with a stack of layers, feeding linearly from one to the next.
 *
 * `tf.sequential` is a factory function that creates an instance of
 * `tf.Sequential`.
 *
 * ```js
 *  // Define a model for linear regression.
 *  const model = tf.sequential();
 *  model.add(tf.layers.dense({units: 1, inputShape: [1]}));
 *
 *  // Prepare the model for training: Specify the loss and the optimizer.
 *  model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});
 *
 *  // Generate some synthetic data for training.
 *  const xs = tf.tensor2d([1, 2, 3, 4], [4, 1]);
 *  const ys = tf.tensor2d([1, 3, 5, 7], [4, 1]);
 *
 *  // Train the model using the data then do inference on a data point the
 *  // model hasn't seen:
 *  await model.fit(xs, ys);
 *  model.predict(tf.tensor2d([5], [1, 1])).print();
 * ```
 *
 * @doc {heading: 'Models', subheading: 'Classes'}
 */
declare class Sequential extends LayersModel {
    /** @nocollapse */
    static className: string;
    private model;
    constructor(args?: SequentialArgs);
    private checkShape;
    /**
     * Adds a layer instance on top of the layer stack.
     *
     * ```js
     *  const model = tf.sequential();
     *  model.add(tf.layers.dense({units: 8, inputShape: [1]}));
     *  model.add(tf.layers.dense({units: 4, activation: 'relu6'}));
     *  model.add(tf.layers.dense({units: 1, activation: 'relu6'}));
     *  // Note that the untrained model is random at this point.
     *  model.predict(tf.randomNormal([10, 1])).print();
     * ```
     * @param layer Layer instance.
     *
     * @exception ValueError In case the `layer` argument does not know its
     * input shape.
     * @exception ValueError In case the `layer` argument has multiple output
     *   tensors, or is already connected somewhere else (forbidden in
     *   `Sequential` models).
     *
     * @doc {heading: 'Models', subheading: 'Classes'}
     */
    add(layer: Layer): void;
    /**
     * Removes the last layer in the model.
     *
     * @exception TypeError if there are no layers in the model.
     */
    pop(): void;
    call(inputs: Tensor | Tensor[], kwargs: Kwargs): Tensor | Tensor[];
    build(inputShape?: Shape | Shape[]): void;
    countParams(): number;
    /**
     * Print a text summary of the Sequential model's layers.
     *
     * The summary includes
     * - Name and type of all layers that comprise the model.
     * - Output shape(s) of the layers
     * - Number of weight parameters of each layer
     * - The total number of trainable and non-trainable parameters of the
     * model.
     *
     * ```js
     * const model = tf.sequential();
     * model.add(
     *     tf.layers.dense({units: 100, inputShape: [10], activation: 'relu'}));
     * model.add(tf.layers.dense({units: 1, activation: 'sigmoid'}));
     *
     * model.summary();
     * ```
     *
     * @param lineLength Custom line length, in number of characters.
     * @param positions Custom widths of each of the columns, as either
     *   fractions of `lineLength` (e.g., `[0.5, 0.75, 1]`) or absolute number
     *   of characters (e.g., `[30, 50, 65]`). Each number corresponds to
     *   right-most (i.e., ending) position of a column.
     * @param printFn Custom print function. Can be used to replace the default
     *   `console.log`. For example, you can use `x => {}` to mute the printed
     *   messages in the console.
     *
     * @doc {heading: 'Models', subheading: 'Classes'}
     */
    summary(lineLength?: number, positions?: number[], printFn?: (message?: any, ...optionalParams: any[]) => void): void;
    /**
     * Sets the weights of the model.
     *
     * @param weights Should be a list of Tensors with shapes and types matching
     *   the output of `model.getWeights()`.
     */
    setWeights(weights: Tensor[]): void;
    /**
     * Returns the loss value & metrics values for the model in test mode.
     *
     * Loss and metrics are specified during `compile()`, which needs to happen
     * before calls to `evaluate()`.
     *
     * Computation is done in batches.
     *
     * ```js
     * const model = tf.sequential({
     *   layers: [tf.layers.dense({units: 1, inputShape: [10]})]
     * });
     * model.compile({optimizer: 'sgd', loss: 'meanSquaredError'});
     * const result = model.evaluate(tf.ones([8, 10]), tf.ones([8, 1]), {
     *   batchSize: 4,
     * });
     * result.print();
     * ```
     *
     * @param x `tf.Tensor` of test data, or an `Array` of `tf.Tensor`s if the
     * model has multiple inputs.
     * @param y `tf.Tensor` of target data, or an `Array` of `tf.Tensor`s if the
     * model has multiple outputs.
     * @param args A `ModelEvaluateConfig`, containing optional fields.
     *
     * @return `Scalar` test loss (if the model has a single output and no
     *   metrics) or `Array` of `Scalar`s (if the model has multiple outputs
     *   and/or metrics). The attribute `model.metricsNames`
     *   will give you the display labels for the scalar outputs.
     *
     * @doc {heading: 'Models', subheading: 'Classes'}
     */
    evaluate(x: Tensor | Tensor[], y: Tensor | Tensor[], args?: ModelEvaluateArgs): Scalar | Scalar[];
    /**
     * Evaluate model using a dataset object.
     *
     * Note: Unlike `evaluate()`, this method is asynchronous (`async`).
     *
     * @param dataset A dataset object. Its `iterator()` method is expected
     *   to generate a dataset iterator object, the `next()` method of which
     *   is expected to produce data batches for evaluation. The return value
     *   of the `next()` call ought to contain a boolean `done` field and a
     *   `value` field. The `value` field is expected to be an array of two
     *   `tf.Tensor`s or an array of two nested `tf.Tensor` structures. The former
     *   case is for models with exactly one input and one output (e.g.
     *   a sequential model). The latter case is for models with multiple
     *   inputs and/or multiple outputs. Of the two items in the array, the
     *   first is the input feature(s) and the second is the output target(s).
     * @param args A configuration object for the dataset-based evaluation.
     * @returns Loss and metric values as an Array of `Scalar` objects.
     *
     * @doc {heading: 'Models', subheading: 'Classes'}
     */
    evaluateDataset(dataset: Dataset<{}>, args: ModelEvaluateDatasetArgs): Promise<Scalar | Scalar[]>;
    /**
     * Generates output predictions for the input samples.
     *
     * Computation is done in batches.
     *
     * Note: the "step" mode of predict() is currently not supported.
     *   This is because the TensorFlow.js core backend is imperative only.
     *
     * ```js
     * const model = tf.sequential({
     *   layers: [tf.layers.dense({units: 1, inputShape: [10]})]
     * });
     * model.predict(tf.ones([2, 10])).print();
     * ```
     *
     * @param x The input data, as a Tensor, or an `Array` of `tf.Tensor`s if
     *   the model has multiple inputs.
     * @param conifg A `ModelPredictConfig` object containing optional fields.
     *
     * @return `tf.Tensor`(s) of predictions.
     *
     * @exception ValueError In case of mismatch between the provided input data
     *   and the model's expectations, or in case a stateful model receives a
     *   number of samples that is not a multiple of the batch size.
     *
     * @doc {heading: 'Models', subheading: 'Classes'}
     */
    predict(x: Tensor | Tensor[], args?: ModelPredictArgs): Tensor | Tensor[];
    /**
     * Returns predictions for a single batch of samples.
     *
     * @param x: Input samples, as a Tensor, or list of Tensors (if the model
     *   has multiple inputs).
     * @return Tensor(s) of predictions
     */
    predictOnBatch(x: Tensor): Tensor | Tensor[];
    /**
     * See `LayersModel.compile`.
     *
     * @param args
     */
    compile(args: ModelCompileArgs): void;
    get optimizer(): Optimizer;
    set optimizer(optimizer: Optimizer);
    /**
     * Trains the model for a fixed number of epochs (iterations on a dataset).
     *
     * ```js
     * const model = tf.sequential({
     *   layers: [tf.layers.dense({units: 1, inputShape: [10]})]
     * });
     * model.compile({optimizer: 'sgd', loss: 'meanSquaredError'});
     * const history = await model.fit(tf.ones([8, 10]), tf.ones([8, 1]), {
     *   batchSize: 4,
     *   epochs: 3
     * });
     * console.log(history.history.loss[0]);
     * ```
     *
     * @param x `tf.Tensor` of training data, or an array of `tf.Tensor`s if the
     * model has multiple inputs. If all inputs in the model are named, you can
     * also pass a dictionary mapping input names to `tf.Tensor`s.
     * @param y `tf.Tensor` of target (label) data, or an array of `tf.Tensor`s if
     * the model has multiple outputs. If all outputs in the model are named, you
     *  can also pass a dictionary mapping output names to `tf.Tensor`s.
     * @param args  A `ModelFitConfig`, containing optional fields.
     *
     * @return A `History` instance. Its `history` attribute contains all
     *   information collected during training.
     *
     * @exception ValueError In case of mismatch between the provided input data
     *   and what the model expects.
     *
     * @doc {heading: 'Models', subheading: 'Classes'}
     */
    fit(x: Tensor | Tensor[] | {
        [inputName: string]: Tensor;
    }, y: Tensor | Tensor[] | {
        [inputName: string]: Tensor;
    }, args?: ModelFitArgs): Promise<History>;
    /**
     * Trains the model using a dataset object.
     *
     * ```js
     * const xArray = [
     *   [1, 1, 1, 1, 1, 1, 1, 1, 1],
     *   [1, 1, 1, 1, 1, 1, 1, 1, 1],
     *   [1, 1, 1, 1, 1, 1, 1, 1, 1],
     *   [1, 1, 1, 1, 1, 1, 1, 1, 1],
     * ];
     * const yArray = [1, 1, 1, 1];
     * // Create a dataset from the JavaScript array.
     * const xDataset = tf.data.array(xArray);
     * const yDataset = tf.data.array(yArray);
     * // Zip combines the `x` and `y` Datasets into a single Dataset, the
     * // iterator of which will return an object containing of two tensors,
     * // corresponding to `x` and `y`.  The call to `batch(4)` will bundle
     * // four such samples into a single object, with the same keys now pointing
     * // to tensors that hold 4 examples, organized along the batch dimension.
     * // The call to `shuffle(4)` causes each iteration through the dataset to
     * // happen in a different order.  The size of the shuffle window is 4.
     * const xyDataset = tf.data.zip({xs: xDataset, ys: yDataset})
     *     .batch(4)
     *     .shuffle(4);
     * const model = tf.sequential({
     *   layers: [tf.layers.dense({units: 1, inputShape: [9]})]
     * });
     * model.compile({optimizer: 'sgd', loss: 'meanSquaredError'});
     * const history = await model.fitDataset(xyDataset, {
     *   epochs: 4,
     *   callbacks: {onEpochEnd: (epoch, logs) => console.log(logs.loss)}
     * });
     * ```
     *
     * @param dataset A dataset object. Its `iterator()` method is expected to
     *   generate a dataset iterator object, the `next()` method of which is
     *   expected to produce data batches for evaluation. The return value of the
     *   `next()` call ought to contain a boolean `done` field and a `value`
     *   field.
     *
     *   The `value` field is expected to be an object of with fields
     *   `xs` and `ys`, which point to the feature tensor and the target tensor,
     *   respectively. This case is for models with exactly one input and one
     *   output (e.g. a sequential model). For example:
     *   ```js
     *   {value: {xs: xsTensor, ys: ysTensor}, done: false}
     *   ```
     *
     *   If the model has multiple inputs, the `xs` field of `value` should
     *   be an object mapping input names to their respective feature tensors.
     *   For example:
     *   ```js
     *   {
     *     value: {
     *       xs: {
     *         input_1: xsTensor1,
     *         input_2: xsTensor2
     *       },
     *       ys: ysTensor
     *     },
     *     done: false
     *   }
     *   ```
     *   If the model has multiple outputs, the `ys` field of `value` should
     *   be an object mapping output names to their respective target tensors.
     *   For example:
     *   ```js
     *   {
     *     value: {
     *       xs: xsTensor,
     *       ys: {
     *         output_1: ysTensor1,
     *         output_2: ysTensor2
     *       },
     *     },
     *     done: false
     *   }
     *   ```
     * @param args A `ModelFitDatasetArgs`, containing optional fields.
     *
     * @return A `History` instance. Its `history` attribute contains all
     *   information collected during training.
     *
     * @doc {heading: 'Models', subheading: 'Classes', ignoreCI: true}
     */
    fitDataset<T>(dataset: Dataset<T>, args: ModelFitDatasetArgs<T>): Promise<History>;
    /**
     * Runs a single gradient update on a single batch of data.
     *
     * This method differs from `fit()` and `fitDataset()` in the following
     * regards:
     *   - It operates on exactly one batch of data.
     *   - It returns only the loss and metric values, instead of
     *     returning the batch-by-batch loss and metric values.
     *   - It doesn't support fine-grained options such as verbosity and
     *     callbacks.
     *
     * @param x Input data. It could be one of the following:
     *   - A `tf.Tensor`, or an Array of `tf.Tensor`s (in case the model has
     *     multiple inputs).
     *   - An Object mapping input names to corresponding `tf.Tensor` (if the
     *     model has named inputs).
     * @param y Target data. It could be either a `tf.Tensor` or multiple
     *   `tf.Tensor`s. It should be consistent with `x`.
     * @returns Training loss or losses (in case the model has
     *   multiple outputs), along with metrics (if any), as numbers.
     *
     * @doc {heading: 'Models', subheading: 'Classes'}
     */
    trainOnBatch(x: Tensor | Tensor[] | {
        [inputName: string]: Tensor;
    }, y: Tensor | Tensor[] | {
        [inputName: string]: Tensor;
    }): Promise<number | number[]>;
    /** @nocollapse */
    static fromConfig<T extends serialization.Serializable>(cls: serialization.SerializableConstructor<T>, config: serialization.ConfigDict, customObjects?: serialization.ConfigDict, fastWeightInit?: boolean): T;
    /**
     * Setter used for force stopping of LayersModel.fit() (i.e., training).
     *
     * Example:
     *
     * ```js
     * const model = tf.sequential();
     * model.add(tf.layers.dense({units: 1, inputShape: [10]}));
     * model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});
     * const xs = tf.ones([8, 10]);
     * const ys = tf.zeros([8, 1]);
     *
     * const history = await model.fit(xs, ys, {
     *   epochs: 10,
     *   callbacks: {
     *     onEpochEnd: async (epoch, logs) => {
     *       if (epoch === 2) {
     *         model.stopTraining = true;
     *       }
     *     }
     *   }
     * });
     *
     * // There should be only 3 values in the loss array, instead of 10 values,
     * // due to the stopping after 3 epochs.
     * console.log(history.history.loss);
     * ```
     */
    set stopTraining(stop: boolean);
    get stopTraining(): boolean;
    getConfig(): any;
}

/// <amd-module name="@tensorflow/tfjs-core/dist/io/model_management" />
declare class ModelStoreManagerRegistry {
    private static instance;
    private managers;
    private constructor();
    private static getInstance;
    /**
     * Register a save-handler router.
     *
     * @param saveRouter A function that maps a URL-like string onto an instance
     * of `IOHandler` with the `save` method defined or `null`.
     */
    static registerManager(scheme: string, manager: ModelStoreManager): void;
    static getManager(scheme: string): ModelStoreManager;
    static getSchemes(): string[];
}
/**
 * List all models stored in registered storage mediums.
 *
 * For a web browser environment, the registered mediums are Local Storage and
 * IndexedDB.
 *
 * ```js
 * // First create and save a model.
 * const model = tf.sequential();
 * model.add(tf.layers.dense(
 *     {units: 1, inputShape: [10], activation: 'sigmoid'}));
 * await model.save('localstorage://demo/management/model1');
 *
 * // Then list existing models.
 * console.log(JSON.stringify(await tf.io.listModels()));
 *
 * // Delete the model.
 * await tf.io.removeModel('localstorage://demo/management/model1');
 *
 * // List models again.
 * console.log(JSON.stringify(await tf.io.listModels()));
 * ```
 *
 * @returns A `Promise` of a dictionary mapping URLs of existing models to
 * their model artifacts info. URLs include medium-specific schemes, e.g.,
 *   'indexeddb://my/model/1'. Model artifacts info include type of the
 * model's topology, byte sizes of the topology, weights, etc.
 *
 * @doc {
 *   heading: 'Models',
 *   subheading: 'Management',
 *   namespace: 'io',
 *   ignoreCI: true
 * }
 */
declare function listModels(): Promise<{
    [url: string]: ModelArtifactsInfo;
}>;
/**
 * Remove a model specified by URL from a registered storage medium.
 *
 * ```js
 * // First create and save a model.
 * const model = tf.sequential();
 * model.add(tf.layers.dense(
 *     {units: 1, inputShape: [10], activation: 'sigmoid'}));
 * await model.save('localstorage://demo/management/model1');
 *
 * // Then list existing models.
 * console.log(JSON.stringify(await tf.io.listModels()));
 *
 * // Delete the model.
 * await tf.io.removeModel('localstorage://demo/management/model1');
 *
 * // List models again.
 * console.log(JSON.stringify(await tf.io.listModels()));
 * ```
 *
 * @param url A URL to a stored model, with a scheme prefix, e.g.,
 *   'localstorage://my-model-1', 'indexeddb://my/model/2'.
 * @returns ModelArtifactsInfo of the deleted model (if and only if deletion
 *   is successful).
 * @throws Error if deletion fails, e.g., if no model exists at `path`.
 *
 * @doc {
 *   heading: 'Models',
 *   subheading: 'Management',
 *   namespace: 'io',
 *   ignoreCI: true
 * }
 */
declare function removeModel(url: string): Promise<ModelArtifactsInfo>;
/**
 * Copy a model from one URL to another.
 *
 * This function supports:
 *
 * 1. Copying within a storage medium, e.g.,
 *    `tf.io.copyModel('localstorage://model-1', 'localstorage://model-2')`
 * 2. Copying between two storage mediums, e.g.,
 *    `tf.io.copyModel('localstorage://model-1', 'indexeddb://model-1')`
 *
 * ```js
 * // First create and save a model.
 * const model = tf.sequential();
 * model.add(tf.layers.dense(
 *     {units: 1, inputShape: [10], activation: 'sigmoid'}));
 * await model.save('localstorage://demo/management/model1');
 *
 * // Then list existing models.
 * console.log(JSON.stringify(await tf.io.listModels()));
 *
 * // Copy the model, from Local Storage to IndexedDB.
 * await tf.io.copyModel(
 *     'localstorage://demo/management/model1',
 *     'indexeddb://demo/management/model1');
 *
 * // List models again.
 * console.log(JSON.stringify(await tf.io.listModels()));
 *
 * // Remove both models.
 * await tf.io.removeModel('localstorage://demo/management/model1');
 * await tf.io.removeModel('indexeddb://demo/management/model1');
 * ```
 *
 * @param sourceURL Source URL of copying.
 * @param destURL Destination URL of copying.
 * @returns ModelArtifactsInfo of the copied model (if and only if copying
 *   is successful).
 * @throws Error if copying fails, e.g., if no model exists at `sourceURL`, or
 *   if `oldPath` and `newPath` are identical.
 *
 * @doc {
 *   heading: 'Models',
 *   subheading: 'Management',
 *   namespace: 'io',
 *   ignoreCI: true
 * }
 */
declare function copyModel(sourceURL: string, destURL: string): Promise<ModelArtifactsInfo>;
/**
 * Move a model from one URL to another.
 *
 * This function supports:
 *
 * 1. Moving within a storage medium, e.g.,
 *    `tf.io.moveModel('localstorage://model-1', 'localstorage://model-2')`
 * 2. Moving between two storage mediums, e.g.,
 *    `tf.io.moveModel('localstorage://model-1', 'indexeddb://model-1')`
 *
 * ```js
 * // First create and save a model.
 * const model = tf.sequential();
 * model.add(tf.layers.dense(
 *     {units: 1, inputShape: [10], activation: 'sigmoid'}));
 * await model.save('localstorage://demo/management/model1');
 *
 * // Then list existing models.
 * console.log(JSON.stringify(await tf.io.listModels()));
 *
 * // Move the model, from Local Storage to IndexedDB.
 * await tf.io.moveModel(
 *     'localstorage://demo/management/model1',
 *     'indexeddb://demo/management/model1');
 *
 * // List models again.
 * console.log(JSON.stringify(await tf.io.listModels()));
 *
 * // Remove the moved model.
 * await tf.io.removeModel('indexeddb://demo/management/model1');
 * ```
 *
 * @param sourceURL Source URL of moving.
 * @param destURL Destination URL of moving.
 * @returns ModelArtifactsInfo of the copied model (if and only if copying
 *   is successful).
 * @throws Error if moving fails, e.g., if no model exists at `sourceURL`, or
 *   if `oldPath` and `newPath` are identical.
 *
 * @doc {
 *   heading: 'Models',
 *   subheading: 'Management',
 *   namespace: 'io',
 *   ignoreCI: true
 * }
 */
declare function moveModel(sourceURL: string, destURL: string): Promise<ModelArtifactsInfo>;
{ moveModel, copyModel, removeModel, listModels };

/// <amd-module name="@tensorflow/tfjs-core/dist/io/model_management_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-layers/dist/keras_format/model_serialization" />
declare type ModelConfig = {
    name: string;
    layers: LayerSerialization[];
    input_layers: TensorKeyArray[];
    output_layers: TensorKeyArray[];
};
/**
 * A standard Keras JSON 'Model' configuration.
 */
interface ModelSerialization extends BaseSerialization<'Model', ModelConfig> {
    backend?: string;
    keras_version?: string;
}
declare type SequentialConfig = {
    layers: LayerSerialization[];
};
/**
 * A standard Keras JSON 'Sequential' configuration.
 */
interface SequentialSerialization extends BaseSerialization<'Sequential', SequentialConfig> {
    backend?: string;
    keras_version?: string;
}
/**
 * A legacy Keras JSON 'Sequential' configuration.
 *
 * It was a bug that Keras Sequential models were recorded with
 * model_config.config as an array of layers, instead of a dict containing a
 * 'layers' entry.  While the bug has been fixed, we still need to be able to
 * read this legacy format.
 */
declare type LegacySequentialSerialization = {
    class_name: 'Sequential';
    config: LayerSerialization[];
    backend?: string;
    keras_version?: string;
};
/**
 * Contains the description of a KerasModel, as well as the configuration
 * necessary to train that model.
 */
declare type KerasFileSerialization = {
    model_config: ModelSerialization | SequentialSerialization | LegacySequentialSerialization;
    training_config: TrainingConfig;
};

/// <amd-module name="@tensorflow/tfjs-core/dist/model_types" />
interface ModelPredictConfig {
    /**
     * Optional. Batch size (Integer). If unspecified, it will default to 32.
     */
    batchSize?: number;
    /**
     * Optional. Verbosity mode. Defaults to false.
     */
    verbose?: boolean;
}
/**
 * Interface for model input/output tensor info.
 */
interface ModelTensorInfo {
    name: string;
    shape?: number[];
    dtype: DataType;
    tfDtype?: string;
}
/**
 * Common interface for a machine learning model that can do inference.
 */
interface InferenceModel {
    /**
     * Return the array of input tensor info.
     */
    readonly inputs: ModelTensorInfo[];
    /**
     * Return the array of output tensor info.
     */
    readonly outputs: ModelTensorInfo[];
    /**
     * Execute the inference for the input tensors.
     *
     * @param input The input tensors, when there is single input for the model,
     * inputs param should be a Tensor. For models with multiple inputs, inputs
     * params should be in either Tensor[] if the input order is fixed, or
     * otherwise NamedTensorMap format.
     * For batch inference execution, the tensors for each input need to be
     * concatenated together. For example with mobilenet, the required input shape
     * is [1, 244, 244, 3], which represents the [batch, height, width, channel].
     * If we are provide a batched data of 100 images, the input tensor should be
     * in the shape of [100, 244, 244, 3].
     *
     * @param config Prediction configuration for specifying the batch size.
     *
     * @returns Inference result tensors. The output would be single Tensor if
     * model has single output node, otherwise Tensor[] or NamedTensorMap[] will
     * be returned for model with multiple outputs.
     */
    predict(inputs: Tensor | Tensor[] | NamedTensorMap, config: ModelPredictConfig): Tensor | Tensor[] | NamedTensorMap;
    /**
     * Single Execute the inference for the input tensors and return activation
     * values for specified output node names without batching.
     *
     * @param input The input tensors, when there is single input for the model,
     * inputs param should be a Tensor. For models with multiple inputs, inputs
     * params should be in either Tensor[] if the input order is fixed, or
     * otherwise NamedTensorMap format.
     *
     * @param outputs string|string[]. List of output node names to retrieve
     * activation from.
     *
     * @returns Activation values for the output nodes result tensors. The return
     * type matches specified parameter outputs type. The output would be single
     * Tensor if single output is specified, otherwise Tensor[] for multiple
     * outputs.
     */
    execute(inputs: Tensor | Tensor[] | NamedTensorMap, outputs: string | string[]): Tensor | Tensor[];
}
/**
 * @deprecated Deprecated interface for SavedModel/GraphModel MetaGraph info.
 *     User MetaGraph instead.
 */
interface MetaGraphInfo {
    tags: string[];
    signatureDefs: SignatureDefInfo;
}
/**
 * @deprecated Deprecated interface for SavedModel/GraphModel SignatureDef info.
 *     User SignatureDef instead.
 */
interface SignatureDefInfo {
    [key: string]: {
        inputs: {
            [key: string]: SavedModelTensorInfo;
        };
        outputs: {
            [key: string]: SavedModelTensorInfo;
        };
    };
}
/**
 * @deprecated Deprecated interface for SavedModel/GraphModel signature
 *     input/output Tensor info. User ModelTensorInfo instead.
 */
interface SavedModelTensorInfo {
    dtype: string;
    shape: number[];
    name: string;
}
/**
 * Interface for SavedModel/GraphModel MetaGraph info.
 */
interface MetaGraph {
    tags: string[];
    signatureDefs: SignatureDef;
}
/**
 * Interface for SavedModel/GraphModel SignatureDef entry.
 */
interface SignatureDefEntry {
    inputs: {
        [key: string]: ModelTensorInfo;
    };
    outputs: {
        [key: string]: ModelTensorInfo;
    };
}
/**
 * Interface for SavedModel/GraphModel SignatureDef info.
 */
interface SignatureDef {
    [key: string]: SignatureDefEntry;
}

/// <amd-module name="@tensorflow/tfjs-core/dist/gradients/Mod_grad" />
declare const modGradConfig: GradConfig;

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/moments" />
/**
 * Calculates the mean and variance of `x`. The mean and variance are
 * calculated by aggregating the contents of `x` across `axes`. If `x` is
 * 1-D and `axes = [0]` this is just the mean and variance of a vector.
 *
 * @param x The input tensor.
 * @param axis The dimension(s) along with to compute mean and
 *     variance. By default it reduces all dimensions.
 * @param keepDims If true, the moments have the same dimensionality as the
 *     input.
 * @return An object with two keys: `mean` and `variance`.
 *
 * @doc {heading: 'Operations', subheading: 'Normalization'}
 */
declare function moments_(x: Tensor | TensorLike, axis?: number | number[], keepDims?: boolean): {
    mean: Tensor;
    variance: Tensor;
};
declare const moments: typeof moments_;
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/moments_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/optimizers/momentum_optimizer" />
/** @doclink Optimizer */
declare class MomentumOptimizer extends SGDOptimizer {
    protected learningRate: number;
    private momentum;
    private useNesterov;
    /** @nocollapse */
    static get className(): string;
    private m;
    private accumulations;
    constructor(learningRate: number, momentum: number, useNesterov?: boolean);
    applyGradients(variableGradients: NamedVariableMap | NamedTensor[]): void;
    dispose(): void;
    /**
     * Sets the momentum of the optimizer.
     *
     * @param momentum
     */
    setMomentum(momentum: number): void;
    getWeights(): Promise<NamedTensor[]>;
    setWeights(weightValues: NamedTensor[]): Promise<void>;
    getConfig(): ConfigDict;
    /** @nocollapse */
    static fromConfig<T extends Serializable>(cls: SerializableConstructor<T>, config: ConfigDict): T;
}

/// <amd-module name="@tensorflow/tfjs-core/dist/optimizers/momentum_optimizer_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/moving_average" />
/**
 * Compute the moving average of a variable.
 *
 * Without zeroDebias, the moving average operation is defined by:
 *   `v += delta`
 * where
 *   `delta = (1 - decay) * (x - v)`
 *
 * With zeroDebias (default), the `delta` term is scaled to debias the
 * effect of the (assumed) zero-initialization of `v`.
 *   `delta /= (1 - decay ^ step)`
 *
 * For more details on the zero-debiasing algorithm, see:
 *   https://arxiv.org/abs/1412.6980
 *
 * Note that this function is completely stateless and does not keep track of
 * step count. The step count needs to be maintained by the caller and passed
 * in as `step`.
 *
 * @param v The current moving average value.
 * @param x New input value, must have the same shape and dtype as `v`.
 * @param decay The decay factor. Typical values are 0.95 and 0.99.
 * @param step Step count.
 * @param zeroDebias: Whether zeroDebias is to be performed (default: `true`).
 * @returns The new moving average value.
 *
 * @doc {heading: 'Operations', subheading: 'Moving Average'}
 */
declare function movingAverage_<T extends Tensor>(v: T | TensorLike, x: T | TensorLike, decay: number | Scalar, step?: number | Scalar, zeroDebias?: boolean): T;
declare const movingAverage: typeof movingAverage_;
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/moving_average_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/mul" />
/**
 * Multiplies two `tf.Tensor`s element-wise, A * B. Supports broadcasting.
 *
 * We also expose `tf.mulStrict` which has the same signature as this op and
 * asserts that `a` and `b` are the same shape (does not broadcast).
 *
 * ```js
 * const a = tf.tensor1d([1, 2, 3, 4]);
 * const b = tf.tensor1d([2, 3, 4, 5]);
 *
 * a.mul(b).print();  // or tf.mul(a, b)
 * ```
 *
 * ```js
 * // Broadcast mul a with b.
 * const a = tf.tensor1d([1, 2, 3, 4]);
 * const b = tf.scalar(5);
 *
 * a.mul(b).print();  // or tf.mul(a, b)
 * ```
 * @param a The first tensor to multiply.
 * @param b The second tensor to multiply. Must have the same dtype as `a`.
 *
 * @doc {heading: 'Operations', subheading: 'Arithmetic'}
 */
declare function mul_<T extends Tensor>(a: Tensor | TensorLike, b: Tensor | TensorLike): T;
declare const mul: typeof mul_;
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/multinomial" />
/**
 * Creates a `tf.Tensor` with values drawn from a multinomial distribution.
 *
 * ```js
 * const probs = tf.tensor([.75, .25]);
 * tf.multinomial(probs, 3).print();
 * ```
 *
 * @param logits 1D array with unnormalized log-probabilities, or
 *     2D array of shape `[batchSize, numOutcomes]`. See the `normalized`
 *     parameter.
 * @param numSamples Number of samples to draw for each row slice.
 * @param seed The seed number.
 * @param normalized Whether the provided `logits` are normalized true
 *     probabilities (sum to 1). Defaults to false.
 * @return 1D array of shape `[numSamples]`, or 2D array of shape
 *     `[batchSize, numSamples]`, depending on the rank of the input.
 *
 * @doc {heading: 'Tensors', subheading: 'Random'}
 */
declare function multinomial_(logits: Tensor1D | Tensor2D | TensorLike, numSamples: number, seed?: number, normalized?: boolean): Tensor1D | Tensor2D;
declare const multinomial: typeof multinomial_;
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/multinomial_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/gradients/Multiply_grad" />
declare const multiplyGradConfig: GradConfig;
/// <amd-module name="@tensorflow/tfjs-core/dist/ops/multi_rnn_cell" />

/**
 * @docalias (data: Tensor2D, c: Tensor2D, h: Tensor2D): [Tensor2D, Tensor2D]
 */
declare type LSTMCellFunc = {
    (data: Tensor2D, c: Tensor2D, h: Tensor2D): [Tensor2D, Tensor2D];
};
/**
 * Computes the next states and outputs of a stack of LSTMCells.
 *
 * Each cell output is used as input to the next cell.
 *
 * Returns `[cellState, cellOutput]`.
 *
 * Derived from tf.contrib.rn.MultiRNNCell.
 *
 * @param lstmCells Array of LSTMCell functions.
 * @param data The input to the cell.
 * @param c Array of previous cell states.
 * @param h Array of previous cell outputs.
 *
 * @doc {heading: 'Operations', subheading: 'RNN'}
 */
declare function multiRNNCell_(lstmCells: LSTMCellFunc[], data: Tensor2D | TensorLike, c: Array<Tensor2D | TensorLike>, h: Array<Tensor2D | TensorLike>): [Tensor2D[], Tensor2D[]];
declare const multiRNNCell: typeof multiRNNCell_;
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/multi_rnn_cell_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/neg" />
/**
 * Computes `-1 * x` element-wise.
 *
 * ```js
 * const x = tf.tensor2d([1, 2, -2, 0], [2, 2]);
 *
 * x.neg().print();  // or tf.neg(x)
 * ```
 *
 * @param x The input tensor.
 *
 * @doc {heading: 'Operations', subheading: 'Basic math'}
 */
declare function neg_<T extends Tensor>(x: T | TensorLike): T;
declare const neg: typeof neg_;
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/gradients/Neg_grad" />
declare const negGradConfig: GradConfig;

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/neg_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-layers/dist/keras_format/node_config" />
/**
 * The unique string name of a Layer.
 */
declare type LayerName = string;
/**
 * The index of a Node, identifying a specific invocation of a given Layer.
 */
declare type NodeIndex = number;
/**
 * The index of a Tensor output by a given Node of a given Layer.
 */
declare type TensorIndex = number;
/**
 * Arguments to the apply(...) method that produced a specific Node.
 */
interface NodeArgs extends PyJsonDict {
}
/**
 * A reference to a specific Tensor, given by its Layer name, Node index, and
 * output index, including the apply() arguments associated with the Node.
 *
 * This is used in `NodeConfig` to specify the inputs to each Node.
 */
declare type TensorKeyWithArgsArray = [
    LayerName,
    NodeIndex,
    TensorIndex,
    NodeArgs
];
/**
 * A reference to a specific Tensor, given by its Layer name, Node index, and
 * output index.
 *
 * This does not include the apply() arguments associated with the Node.  It is
 * used in the LayersModel config to specify the inputLayers and outputLayers.
 * It seems to be an idiosyncrasy of Python Keras that the node arguments are
 * not included here.
 */
declare type TensorKeyArray = [LayerName, NodeIndex, TensorIndex];
/**
 * A Keras JSON entry representing a Node, i.e. a specific instance of a Layer.
 *
 * By Keras JSON convention, a Node is specified as an array of Tensor keys
 * (i.e., references to Tensors output by other Layers) providing the inputs to
 * this Layer in order.
 */
declare type NodeConfig = TensorKeyWithArgsArray[];

/// <amd-module name="@tensorflow/tfjs-layers/dist/layers/noise" />
/**
 * TensorFlow.js Layers: Noise Layers.
 */
declare interface GaussianNoiseArgs extends LayerArgs {
    /** Standard Deviation.  */
    stddev: number;
}
declare class GaussianNoise extends Layer {
    /** @nocollapse */
    static className: string;
    readonly stddev: number;
    constructor(args: GaussianNoiseArgs);
    computeOutputShape(inputShape: Shape | Shape[]): Shape | Shape[];
    getConfig(): {
        stddev: number;
    };
    call(inputs: Tensor | Tensor[], kwargs: Kwargs): Tensor | Tensor[];
}
declare interface GaussianDropoutArgs extends LayerArgs {
    /** drop probability.  */
    rate: number;
}
declare class GaussianDropout extends Layer {
    /** @nocollapse */
    static className: string;
    readonly rate: number;
    constructor(args: GaussianDropoutArgs);
    computeOutputShape(inputShape: Shape | Shape[]): Shape | Shape[];
    getConfig(): {
        rate: number;
    };
    call(inputs: Tensor | Tensor[], kwargs: Kwargs): Tensor | Tensor[];
}
declare interface AlphaDropoutArgs extends LayerArgs {
    /** drop probability.  */
    rate: number;
    /**
     * A 1-D `Tensor` of type `int32`, representing the
     * shape for randomly generated keep/drop flags.
     */
    noiseShape?: Shape;
}
/**
 * Applies Alpha Dropout to the input.
 *
 * As it is a regularization layer, it is only active at training time.
 *
 * Alpha Dropout is a `Dropout` that keeps mean and variance of inputs
 * to their original values, in order to ensure the self-normalizing property
 * even after this dropout.
 * Alpha Dropout fits well to Scaled Exponential Linear Units
 * by randomly setting activations to the negative saturation value.
 *
 * Arguments:
 *   - `rate`: float, drop probability (as with `Dropout`).
 *     The multiplicative noise will have
 *     standard deviation `sqrt(rate / (1 - rate))`.
 *   - `noise_shape`: A 1-D `Tensor` of type `int32`, representing the
 *     shape for randomly generated keep/drop flags.
 *
 * Input shape:
 *   Arbitrary. Use the keyword argument `inputShape`
 *   (tuple of integers, does not include the samples axis)
 *   when using this layer as the first layer in a model.
 *
 * Output shape:
 *   Same shape as input.
 *
 * References:
 *   - [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)
 */
declare class AlphaDropout extends Layer {
    /** @nocollapse */
    static className: string;
    readonly rate: number;
    readonly noiseShape: Shape;
    constructor(args: AlphaDropoutArgs);
    _getNoiseShape(inputs: Tensor | Tensor[]): Shape;
    computeOutputShape(inputShape: Shape | Shape[]): Shape | Shape[];
    getConfig(): {
        rate: number;
    };
    call(inputs: Tensor | Tensor[], kwargs: Kwargs): Tensor | Tensor[];
}

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/nonmax_util" />
declare function nonMaxSuppSanityCheck(boxes: Tensor2D, scores: Tensor1D, maxOutputSize: number, iouThreshold: number, scoreThreshold: number, softNmsSigma?: number): {
    maxOutputSize: number;
    iouThreshold: number;
    scoreThreshold: number;
    softNmsSigma: number;
};
{ nonMaxSuppSanityCheck };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/image/non_max_suppression" />
/**
 * Performs non maximum suppression of bounding boxes based on
 * iou (intersection over union).
 *
 * @param boxes a 2d tensor of shape `[numBoxes, 4]`. Each entry is
 *     `[y1, x1, y2, x2]`, where `(y1, x1)` and `(y2, x2)` are the corners of
 *     the bounding box.
 * @param scores a 1d tensor providing the box scores of shape `[numBoxes]`.
 * @param maxOutputSize The maximum number of boxes to be selected.
 * @param iouThreshold A float representing the threshold for deciding whether
 *     boxes overlap too much with respect to IOU. Must be between [0, 1].
 *     Defaults to 0.5 (50% box overlap).
 * @param scoreThreshold A threshold for deciding when to remove boxes based
 *     on score. Defaults to -inf, which means any score is accepted.
 * @return A 1D tensor with the selected box indices.
 *
 * @doc {heading: 'Operations', subheading: 'Images', namespace: 'image'}
 */
declare function nonMaxSuppression_(boxes: Tensor2D | TensorLike, scores: Tensor1D | TensorLike, maxOutputSize: number, iouThreshold?: number, scoreThreshold?: number): Tensor1D;
declare const nonMaxSuppression: typeof nonMaxSuppression_;
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/image/non_max_suppression_async" />
/**
 * Performs non maximum suppression of bounding boxes based on
 * iou (intersection over union).
 *
 * This is the async version of `nonMaxSuppression`
 *
 * @param boxes a 2d tensor of shape `[numBoxes, 4]`. Each entry is
 *     `[y1, x1, y2, x2]`, where `(y1, x1)` and `(y2, x2)` are the corners of
 *     the bounding box.
 * @param scores a 1d tensor providing the box scores of shape `[numBoxes]`.
 * @param maxOutputSize The maximum number of boxes to be selected.
 * @param iouThreshold A float representing the threshold for deciding whether
 *     boxes overlap too much with respect to IOU. Must be between [0, 1].
 *     Defaults to 0.5 (50% box overlap).
 * @param scoreThreshold A threshold for deciding when to remove boxes based
 *     on score. Defaults to -inf, which means any score is accepted.
 * @return A 1D tensor with the selected box indices.
 *
 * @doc {heading: 'Operations', subheading: 'Images', namespace: 'image'}
 */
declare function nonMaxSuppressionAsync_(boxes: Tensor2D | TensorLike, scores: Tensor1D | TensorLike, maxOutputSize: number, iouThreshold?: number, scoreThreshold?: number): Promise<Tensor1D>;
declare const nonMaxSuppressionAsync: typeof nonMaxSuppressionAsync_;
{ };
/// <amd-module name="@tensorflow/tfjs-core/dist/ops/image/non_max_suppression_async_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/backends/non_max_suppression_impl" />
interface NonMaxSuppressionResult {
    selectedIndices: number[];
    selectedScores?: number[];
    validOutputs?: number;
}
declare function nonMaxSuppressionV3Impl(boxes: TypedArray, scores: TypedArray, maxOutputSize: number, iouThreshold: number, scoreThreshold: number): NonMaxSuppressionResult;
declare function nonMaxSuppressionV4Impl(boxes: TypedArray, scores: TypedArray, maxOutputSize: number, iouThreshold: number, scoreThreshold: number, padToMaxOutputSize: boolean): NonMaxSuppressionResult;
declare function nonMaxSuppressionV5Impl(boxes: TypedArray, scores: TypedArray, maxOutputSize: number, iouThreshold: number, scoreThreshold: number, softNmsSigma: number): NonMaxSuppressionResult;
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/image/non_max_suppression_padded" />
/**
 * Asynchronously performs non maximum suppression of bounding boxes based on
 * iou (intersection over union), with an option to pad results.
 *
 * @param boxes a 2d tensor of shape `[numBoxes, 4]`. Each entry is
 *     `[y1, x1, y2, x2]`, where `(y1, x1)` and `(y2, x2)` are the corners of
 *     the bounding box.
 * @param scores a 1d tensor providing the box scores of shape `[numBoxes]`.
 * @param maxOutputSize The maximum number of boxes to be selected.
 * @param iouThreshold A float representing the threshold for deciding whether
 *     boxes overlap too much with respect to IOU. Must be between [0, 1].
 *     Defaults to 0.5 (50% box overlap).
 * @param scoreThreshold A threshold for deciding when to remove boxes based
 *     on score. Defaults to -inf, which means any score is accepted.
 * @param padToMaxOutputSize Defaults to false. If true, size of output
 *     `selectedIndices` is padded to maxOutputSize.
 * @return A map with the following properties:
 *     - selectedIndices: A 1D tensor with the selected box indices.
 *     - validOutputs: A scalar denoting how many elements in `selectedIndices`
 *       are valid. Valid elements occur first, then padding.
 *
 * @doc {heading: 'Operations', subheading: 'Images', namespace: 'image'}
 */
declare function nonMaxSuppressionPadded_(boxes: Tensor2D | TensorLike, scores: Tensor1D | TensorLike, maxOutputSize: number, iouThreshold?: number, scoreThreshold?: number, padToMaxOutputSize?: boolean): NamedTensorMap;
declare const nonMaxSuppressionPadded: typeof nonMaxSuppressionPadded_;
{ };
/// <amd-module name="@tensorflow/tfjs-core/dist/ops/image/non_max_suppression_padded_async" />
/**
 * Asynchronously performs non maximum suppression of bounding boxes based on
 * iou (intersection over union), with an option to pad results.
 *
 * @param boxes a 2d tensor of shape `[numBoxes, 4]`. Each entry is
 *     `[y1, x1, y2, x2]`, where `(y1, x1)` and `(y2, x2)` are the corners of
 *     the bounding box.
 * @param scores a 1d tensor providing the box scores of shape `[numBoxes]`.
 * @param maxOutputSize The maximum number of boxes to be selected.
 * @param iouThreshold A float representing the threshold for deciding whether
 *     boxes overlap too much with respect to IOU. Must be between [0, 1].
 *     Defaults to 0.5 (50% box overlap).
 * @param scoreThreshold A threshold for deciding when to remove boxes based
 *     on score. Defaults to -inf, which means any score is accepted.
 * @param padToMaxOutputSize Defaults to false. If true, size of output
 *     `selectedIndices` is padded to maxOutputSize.
 * @return A map with the following properties:
 *     - selectedIndices: A 1D tensor with the selected box indices.
 *     - validOutputs: A scalar denoting how many elements in `selectedIndices`
 *       are valid. Valid elements occur first, then padding.
 *
 * @doc {heading: 'Operations', subheading: 'Images', namespace: 'image'}
 */
declare function nonMaxSuppressionPaddedAsync_(boxes: Tensor2D | TensorLike, scores: Tensor1D | TensorLike, maxOutputSize: number, iouThreshold?: number, scoreThreshold?: number, padToMaxOutputSize?: boolean): Promise<NamedTensorMap>;
declare const nonMaxSuppressionPaddedAsync: typeof nonMaxSuppressionPaddedAsync_;
{ };
/// <amd-module name="@tensorflow/tfjs-core/dist/ops/image/non_max_suppression_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/backends/non_max_suppression_util" />
/**
 * Inserts a value into a sorted array. This method allows duplicate, meaning it
 * allows inserting duplicate value, in which case, the element will be inserted
 * at the lowest index of the value.
 * @param arr The array to modify.
 * @param element The element to insert.
 * @param comparator Optional. If no comparator is specified, elements are
 * compared using array_util.defaultComparator, which is suitable for Strings
 * and Numbers in ascending arrays. If the array contains multiple instances of
 * the target value, the left-most instance will be returned. To provide a
 * comparator, it should take 2 arguments to compare and return a negative,
 * zero, or a positive number.
 */
declare function binaryInsert<T>(arr: T[], element: T, comparator?: (a: T, b: T) => number): void;
/**
 * Searches the array for the target using binary search, returns the index
 * of the found element, or position to insert if element not found. If no
 * comparator is specified, elements are compared using array_
 * util.defaultComparator, which is suitable for Strings and Numbers in
 * ascending arrays. If the array contains multiple instances of the target
 * value, the left-most instance will be returned.
 * @param arr The array to be searched in.
 * @param target The target to be searched for.
 * @param comparator Should take 2 arguments to compare and return a negative,
 *    zero, or a positive number.
 * @return Lowest index of the target value if found, otherwise the insertion
 *    point where the target should be inserted, in the form of
 *    (-insertionPoint - 1).
 */
declare function binarySearch<T>(arr: T[], target: T, comparator?: (a: T, b: T) => number): number;

/// <amd-module name="@tensorflow/tfjs-core/dist/backends/non_max_suppression_util_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/image/non_max_suppression_with_score" />
/**
 * Performs non maximum suppression of bounding boxes based on
 * iou (intersection over union).
 *
 * This op also supports a Soft-NMS mode (cf.
 * Bodla et al, https://arxiv.org/abs/1704.04503) where boxes reduce the score
 * of other overlapping boxes, therefore favoring different regions of the image
 * with high scores. To enable this Soft-NMS mode, set the `softNmsSigma`
 * parameter to be larger than 0.
 *
 * @param boxes a 2d tensor of shape `[numBoxes, 4]`. Each entry is
 *     `[y1, x1, y2, x2]`, where `(y1, x1)` and `(y2, x2)` are the corners of
 *     the bounding box.
 * @param scores a 1d tensor providing the box scores of shape `[numBoxes]`.
 * @param maxOutputSize The maximum number of boxes to be selected.
 * @param iouThreshold A float representing the threshold for deciding whether
 *     boxes overlap too much with respect to IOU. Must be between [0, 1].
 *     Defaults to 0.5 (50% box overlap).
 * @param scoreThreshold A threshold for deciding when to remove boxes based
 *     on score. Defaults to -inf, which means any score is accepted.
 * @param softNmsSigma A float representing the sigma parameter for Soft NMS.
 *     When sigma is 0, it falls back to nonMaxSuppression.
 * @return A map with the following properties:
 *     - selectedIndices: A 1D tensor with the selected box indices.
 *     - selectedScores: A 1D tensor with the corresponding scores for each
 *       selected box.
 *
 * @doc {heading: 'Operations', subheading: 'Images', namespace: 'image'}
 */
declare function nonMaxSuppressionWithScore_(boxes: Tensor2D | TensorLike, scores: Tensor1D | TensorLike, maxOutputSize: number, iouThreshold?: number, scoreThreshold?: number, softNmsSigma?: number): NamedTensorMap;
declare const nonMaxSuppressionWithScore: typeof nonMaxSuppressionWithScore_;
{ };
/// <amd-module name="@tensorflow/tfjs-core/dist/ops/image/non_max_suppression_with_score_async" />
/**
 * Asynchronously performs non maximum suppression of bounding boxes based on
 * iou (intersection over union).
 *
 * This op also supports a Soft-NMS mode (cf.
 * Bodla et al, https://arxiv.org/abs/1704.04503) where boxes reduce the score
 * of other overlapping boxes, therefore favoring different regions of the image
 * with high scores. To enable this Soft-NMS mode, set the `softNmsSigma`
 * parameter to be larger than 0.
 *
 * @param boxes a 2d tensor of shape `[numBoxes, 4]`. Each entry is
 *     `[y1, x1, y2, x2]`, where `(y1, x1)` and `(y2, x2)` are the corners of
 *     the bounding box.
 * @param scores a 1d tensor providing the box scores of shape `[numBoxes]`.
 * @param maxOutputSize The maximum number of boxes to be selected.
 * @param iouThreshold A float representing the threshold for deciding whether
 *     boxes overlap too much with respect to IOU. Must be between [0, 1].
 *     Defaults to 0.5 (50% box overlap).
 * @param scoreThreshold A threshold for deciding when to remove boxes based
 *     on score. Defaults to -inf, which means any score is accepted.
 * @param softNmsSigma A float representing the sigma parameter for Soft NMS.
 *     When sigma is 0, it falls back to nonMaxSuppression.
 * @return A map with the following properties:
 *     - selectedIndices: A 1D tensor with the selected box indices.
 *     - selectedScores: A 1D tensor with the corresponding scores for each
 *       selected box.
 *
 * @doc {heading: 'Operations', subheading: 'Images', namespace: 'image'}
 */
declare function nonMaxSuppressionWithScoreAsync_(boxes: Tensor2D | TensorLike, scores: Tensor1D | TensorLike, maxOutputSize: number, iouThreshold?: number, scoreThreshold?: number, softNmsSigma?: number): Promise<NamedTensorMap>;
declare const nonMaxSuppressionWithScoreAsync: typeof nonMaxSuppressionWithScoreAsync_;
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/public/chained_ops/norm" />
declare module '../../tensor' {
    interface Tensor<R extends Rank = Rank> {
        norm<T extends Tensor>(ord?: number | 'euclidean' | 'fro', axis?: number | number[], keepDims?: boolean): Tensor;
    }
}

/// <amd-module name="@tensorflow/tfjs-layers/dist/layers/normalization" />
/**
 * Applies batch normalization on x given mean, var, beta and gamma.
 *
 * I.e. returns:
 *   `output = (x - mean) / (sqrt(var) + epsilon) * gamma + beta`
 *
 * @param x Input tensor.
 * @param mean Mean of batch.
 * @param variance Variance of batch.
 * @param beta Tensor with which to center the input.
 * @param gamma Tensor by which to scale the input.
 * @param epsilon Fuzz factor.
 * @returns The result of the batch normalization.
 */
declare function batchNormalization(x: Tensor, mean: Tensor, variance: Tensor, beta?: Tensor, gamma?: Tensor, epsilon?: number): Tensor;
/**
 * Batch normalization for use in training (not inference).
 *
 * @param x Input tensor to be normalized.
 * @param gamma Tensor by which to scale the input.
 * @param beta Tensor by which to center the input.
 * @param reductionAxes Axes over which to normalize.
 * @param epsilon Fuzz factor.
 * @returns An `Array` of three `Tensors`:
 *   [normalized tensor, mean of input, variance of input].
 */
declare function normalizeBatchInTraining(x: Tensor, gamma: Tensor, beta: Tensor, reductionAxes: number[], epsilon?: number): [Tensor, Tensor, Tensor];
declare interface BatchNormalizationLayerArgs extends LayerArgs {
    /**
     * The integer axis that should be normalized (typically the features axis).
     * Defaults to -1.
     *
     * For instance, after a `Conv2D` layer with `data_format="channels_first"`,
     * set `axis=1` in `batchNormalization`.
     */
    axis?: number;
    /**
     * Momentum of the moving average. Defaults to 0.99.
     */
    momentum?: number;
    /**
     * Small float added to the variance to avoid dividing by zero. Defaults to
     * 1e-3.
     */
    epsilon?: number;
    /**
     * If `true`, add offset of `beta` to normalized tensor.
     * If `false`, `beta` is ignored.
     * Defaults to `true`.
     */
    center?: boolean;
    /**
     * If `true`, multiply by `gamma`.
     * If `false`, `gamma` is not used.
     * When the next layer is linear (also e.g. `nn.relu`),
     * this can be disabled since the scaling will be done by the next layer.
     * Defaults to `true`.
     */
    scale?: boolean;
    /**
     * Initializer for the beta weight.
     *  Defaults to 'zeros'.
     */
    betaInitializer?: InitializerIdentifier | Initializer;
    /**
     * Initializer for the gamma weight.
     *  Defaults to `ones`.
     */
    gammaInitializer?: InitializerIdentifier | Initializer;
    /**
     * Initializer for the moving mean.
     * Defaults to `zeros`
     */
    movingMeanInitializer?: InitializerIdentifier | Initializer;
    /**
     * Initializer for the moving variance.
     *  Defaults to 'Ones'.
     */
    movingVarianceInitializer?: InitializerIdentifier | Initializer;
    /**
     * Constraint for the beta weight.
     */
    betaConstraint?: ConstraintIdentifier | Constraint;
    /**
     * Constraint for gamma weight.
     */
    gammaConstraint?: ConstraintIdentifier | Constraint;
    /**
     * Regularizer for the beta weight.
     */
    betaRegularizer?: RegularizerIdentifier | Regularizer;
    /**
     * Regularizer for the gamma weight.
     */
    gammaRegularizer?: RegularizerIdentifier | Regularizer;
}
declare class BatchNormalization extends Layer {
    /** @nocollapse */
    static className: string;
    private readonly axis;
    private readonly momentum;
    private readonly epsilon;
    private readonly center;
    private readonly scale;
    private readonly betaInitializer;
    private readonly gammaInitializer;
    private readonly movingMeanInitializer;
    private readonly movingVarianceInitializer;
    private readonly betaConstraint;
    private readonly gammaConstraint;
    private readonly betaRegularizer;
    private readonly gammaRegularizer;
    private gamma;
    private beta;
    private movingMean;
    private movingVariance;
    constructor(args?: BatchNormalizationLayerArgs);
    build(inputShape: Shape | Shape[]): void;
    call(inputs: Tensor | Tensor[], kwargs: Kwargs): Tensor | Tensor[];
    getConfig(): serialization.ConfigDict;
}
interface LayerNormalizationLayerArgs extends LayerArgs {
    /**
     * The axis or axes that should be normalized (typically, the feature axis).
     * Defaults to -1 (the last axis).
     */
    axis?: number | number[];
    /**
     * A small positive float added to variance to avoid divison by zero.
     * Defaults to 1e-3.
     */
    epsilon?: number;
    /**
     * If `true`, add offset of `beta` to normalized tensor.
     * If `false`, `beta` is ignored.
     * Default: `true`.
     */
    center?: boolean;
    /**
     * If `true`, multiply output by `gamma`.
     * If `false`, `gamma` is not used.
     * When the next layer is linear, this can be disabled since scaling will
     * be done by the next layer.
     * Default: `true`.
     */
    scale?: boolean;
    /**
     * Initializer for the beta weight.
     * Default: `'zeros'`.
     */
    betaInitializer?: InitializerIdentifier | Initializer;
    /**
     * Initializer for the gamma weight.
     * Default: `'ones'`.
     */
    gammaInitializer?: InitializerIdentifier | Initializer;
    /** Regularizer for the beta weight. */
    betaRegularizer?: RegularizerIdentifier | Regularizer;
    /** Regularizer for the gamma weight. */
    gammaRegularizer?: RegularizerIdentifier | Regularizer;
}
declare class LayerNormalization extends Layer {
    /** @nocollapse */
    static className: string;
    private axis;
    readonly epsilon: number;
    readonly center: boolean;
    readonly scale: boolean;
    readonly betaInitializer: Initializer;
    readonly gammaInitializer: Initializer;
    readonly betaRegularizer: Regularizer;
    readonly gammaRegularizer: Regularizer;
    private gamma;
    private beta;
    constructor(args?: LayerNormalizationLayerArgs);
    build(inputShape: Shape | Shape[]): void;
    call(inputs: Tensor | Tensor[], kwargs: Kwargs): Tensor | Tensor[];
    getConfig(): serialization.ConfigDict;
}

/// <amd-module name="@tensorflow/tfjs-layers/dist/keras_format/layers/normalization_serialization" />
interface BatchNormalizationLayerConfig extends LayerConfig {
    axis?: number;
    momentum?: number;
    epsilon?: number;
    center?: boolean;
    scale?: boolean;
    beta_initializer?: InitializerSerialization;
    gamma_initializer?: InitializerSerialization;
    moving_mean_initializer?: InitializerSerialization;
    moving_variance_initializer?: InitializerSerialization;
    beta_constraint?: ConstraintSerialization;
    gamma_constraint?: ConstraintSerialization;
    beta_regularizer?: RegularizerSerialization;
    gamma_regularizer?: RegularizerSerialization;
}
declare type BatchNormalizationLayerSerialization = BaseLayerSerialization<'BatchNormalization', BatchNormalizationLayerConfig>;
declare type NormalizationLayerSerialization = BatchNormalizationLayerSerialization;
declare type NormalizationLayerClassName = NormalizationLayerSerialization['class_name'];
/**
 * A string array of valid NormalizationLayer class names.
 *
 * This is guaranteed to match the `NormalizationLayerClassName` union
 * type.
 */
declare const normalizationLayerClassNames: NormalizationLayerClassName[];

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/norm_test" />
{ };
/// <amd-module name="@tensorflow/tfjs-core/dist/public/chained_ops/not_equal" />
declare module '../../tensor' {
    interface Tensor<R extends Rank = Rank> {
        notEqual<T extends Tensor>(b: Tensor | TensorLike): T;
    }
}
/// <amd-module name="@tensorflow/tfjs-core/dist/ops/not_equal_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/gradients/OneHot_grad" />
declare const oneHotGradConfig: GradConfig;

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/ones" />
/**
 * Creates a `tf.Tensor` with all elements set to 1.
 *
 * ```js
 * tf.ones([2, 2]).print();
 * ```
 *
 * @param shape An array of integers defining the output tensor shape.
 * @param dtype The type of an element in the resulting tensor. Defaults to
 *     'float'.
 *
 * @doc {heading: 'Tensors', subheading: 'Creation'}
 */
declare function ones<R extends Rank>(shape: ShapeMap[R], dtype?: DataType): Tensor<R>;

/// <amd-module name="@tensorflow/tfjs-core/dist/gradients/OnesLike_grad" />
declare const onesLikeGradConfig: GradConfig;

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/ones_like" />
/**
 * Creates a `tf.Tensor` with all elements set to 1 with the same shape as the
 * given tensor.
 *
 * ```js
 * const x = tf.tensor([1, 2]);
 * tf.onesLike(x).print();
 * ```
 * @param x A tensor.
 *
 * @doc {heading: 'Tensors', subheading: 'Creation'}
 */
declare function onesLike_<T extends Tensor>(x: T | TensorLike): T;
declare const onesLike: typeof onesLike_;
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/ones_like_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/ones_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/public/chained_ops/one_hot" />
declare module '../../tensor' {
    interface Tensor<R extends Rank = Rank> {
        oneHot(depth: number, onValue: number, offValue: number): Tensor;
    }
}

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/one_hot_test" />
{ };
/// <amd-module name="@tensorflow/tfjs-core/dist/ops/operation" />
declare const OP_SCOPE_SUFFIX = "__op";
/**
 * Used for wrapping functions that perform math operations on
 * Tensors. The function will be wrapped in a named scope that cleans all
 * memory usage after the function is done.
 */
declare function op<T extends Function>(f: {
    [name: string]: T;
}): T;
/// <amd-module name="@tensorflow/tfjs-core/dist/ops/operation_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/ops" />
declare const spectral: {
    fft: (input: import("@tensorflow/tfjs-core/dist/tensor").Tensor<import("@tensorflow/tfjs-core/dist/types").Rank>) => import("@tensorflow/tfjs-core/dist/tensor").Tensor<import("@tensorflow/tfjs-core/dist/types").Rank>;
    ifft: (input: import("@tensorflow/tfjs-core/dist/tensor").Tensor<import("@tensorflow/tfjs-core/dist/types").Rank>) => import("@tensorflow/tfjs-core/dist/tensor").Tensor<import("@tensorflow/tfjs-core/dist/types").Rank>;
    rfft: (input: import("@tensorflow/tfjs-core/dist/tensor").Tensor<import("@tensorflow/tfjs-core/dist/types").Rank>, fftLength?: number) => import("@tensorflow/tfjs-core/dist/tensor").Tensor<import("@tensorflow/tfjs-core/dist/types").Rank>;
    irfft: (input: import("@tensorflow/tfjs-core/dist/tensor").Tensor<import("@tensorflow/tfjs-core/dist/types").Rank>) => import("@tensorflow/tfjs-core/dist/tensor").Tensor<import("@tensorflow/tfjs-core/dist/types").Rank>;
};
declare const signal: {
    hammingWindow: (windowLength: number) => import("@tensorflow/tfjs-core/dist/tensor").Tensor1D;
    hannWindow: (windowLength: number) => import("@tensorflow/tfjs-core/dist/tensor").Tensor1D;
    frame: (signal: import("@tensorflow/tfjs-core/dist/tensor").Tensor1D, frameLength: number, frameStep: number, padEnd?: boolean, padValue?: number) => import("@tensorflow/tfjs-core/dist/tensor").Tensor<import("@tensorflow/tfjs-core/dist/types").Rank>;
    stft: (signal: import("@tensorflow/tfjs-core/dist/tensor").Tensor1D, frameLength: number, frameStep: number, fftLength?: number, windowFn?: (length: number) => import("@tensorflow/tfjs-core/dist/tensor").Tensor1D) => import("@tensorflow/tfjs-core/dist/tensor").Tensor<import("@tensorflow/tfjs-core/dist/types").Rank>;
};
declare const image: {
    flipLeftRight: (image: import("@tensorflow/tfjs-core/dist/types").TensorLike | import("@tensorflow/tfjs-core/dist/tensor").Tensor4D) => import("@tensorflow/tfjs-core/dist/tensor").Tensor4D;
    grayscaleToRGB: <T extends import("@tensorflow/tfjs-core/dist/tensor").Tensor2D | import("@tensorflow/tfjs-core/dist/tensor").Tensor3D | import("@tensorflow/tfjs-core/dist/tensor").Tensor4D | import("@tensorflow/tfjs-core/dist/tensor").Tensor5D | import("@tensorflow/tfjs-core/dist/tensor").Tensor6D>(image: import("@tensorflow/tfjs-core/dist/types").TensorLike | T) => T;
    resizeNearestNeighbor: <T_1 extends import("@tensorflow/tfjs-core/dist/tensor").Tensor3D | import("@tensorflow/tfjs-core/dist/tensor").Tensor4D>(images: import("@tensorflow/tfjs-core/dist/types").TensorLike | T_1, size: [number, number], alignCorners?: boolean, halfPixelCenters?: boolean) => T_1;
    resizeBilinear: <T_2 extends import("@tensorflow/tfjs-core/dist/tensor").Tensor3D | import("@tensorflow/tfjs-core/dist/tensor").Tensor4D>(images: import("@tensorflow/tfjs-core/dist/types").TensorLike | T_2, size: [number, number], alignCorners?: boolean, halfPixelCenters?: boolean) => T_2;
    rotateWithOffset: (image: import("@tensorflow/tfjs-core/dist/types").TensorLike | import("@tensorflow/tfjs-core/dist/tensor").Tensor4D, radians: number, fillValue?: number | [number, number, number], center?: number | [number, number]) => import("@tensorflow/tfjs-core/dist/tensor").Tensor4D;
    cropAndResize: (image: import("@tensorflow/tfjs-core/dist/types").TensorLike | import("@tensorflow/tfjs-core/dist/tensor").Tensor4D, boxes: import("@tensorflow/tfjs-core/dist/types").TensorLike | import("@tensorflow/tfjs-core/dist/tensor").Tensor2D, boxInd: import("@tensorflow/tfjs-core/dist/types").TensorLike | import("@tensorflow/tfjs-core/dist/tensor").Tensor1D, cropSize: [number, number], method?: "bilinear" | "nearest", extrapolationValue?: number) => import("@tensorflow/tfjs-core/dist/tensor").Tensor4D;
    nonMaxSuppression: (boxes: import("@tensorflow/tfjs-core/dist/types").TensorLike | import("@tensorflow/tfjs-core/dist/tensor").Tensor2D, scores: import("@tensorflow/tfjs-core/dist/types").TensorLike | import("@tensorflow/tfjs-core/dist/tensor").Tensor1D, maxOutputSize: number, iouThreshold?: number, scoreThreshold?: number) => import("@tensorflow/tfjs-core/dist/tensor").Tensor1D;
    nonMaxSuppressionAsync: (boxes: import("@tensorflow/tfjs-core/dist/types").TensorLike | import("@tensorflow/tfjs-core/dist/tensor").Tensor2D, scores: import("@tensorflow/tfjs-core/dist/types").TensorLike | import("@tensorflow/tfjs-core/dist/tensor").Tensor1D, maxOutputSize: number, iouThreshold?: number, scoreThreshold?: number) => Promise<import("@tensorflow/tfjs-core/dist/tensor").Tensor1D>;
    nonMaxSuppressionWithScore: (boxes: import("@tensorflow/tfjs-core/dist/types").TensorLike | import("@tensorflow/tfjs-core/dist/tensor").Tensor2D, scores: import("@tensorflow/tfjs-core/dist/types").TensorLike | import("@tensorflow/tfjs-core/dist/tensor").Tensor1D, maxOutputSize: number, iouThreshold?: number, scoreThreshold?: number, softNmsSigma?: number) => import("@tensorflow/tfjs-core/dist/tensor_types").NamedTensorMap;
    nonMaxSuppressionWithScoreAsync: (boxes: import("@tensorflow/tfjs-core/dist/types").TensorLike | import("@tensorflow/tfjs-core/dist/tensor").Tensor2D, scores: import("@tensorflow/tfjs-core/dist/types").TensorLike | import("@tensorflow/tfjs-core/dist/tensor").Tensor1D, maxOutputSize: number, iouThreshold?: number, scoreThreshold?: number, softNmsSigma?: number) => Promise<import("@tensorflow/tfjs-core/dist/tensor_types").NamedTensorMap>;
    nonMaxSuppressionPadded: (boxes: import("@tensorflow/tfjs-core/dist/types").TensorLike | import("@tensorflow/tfjs-core/dist/tensor").Tensor2D, scores: import("@tensorflow/tfjs-core/dist/types").TensorLike | import("@tensorflow/tfjs-core/dist/tensor").Tensor1D, maxOutputSize: number, iouThreshold?: number, scoreThreshold?: number, padToMaxOutputSize?: boolean) => import("@tensorflow/tfjs-core/dist/tensor_types").NamedTensorMap;
    nonMaxSuppressionPaddedAsync: (boxes: import("@tensorflow/tfjs-core/dist/types").TensorLike | import("@tensorflow/tfjs-core/dist/tensor").Tensor2D, scores: import("@tensorflow/tfjs-core/dist/types").TensorLike | import("@tensorflow/tfjs-core/dist/tensor").Tensor1D, maxOutputSize: number, iouThreshold?: number, scoreThreshold?: number, padToMaxOutputSize?: boolean) => Promise<import("@tensorflow/tfjs-core/dist/tensor_types").NamedTensorMap>;
    threshold: (image: import("@tensorflow/tfjs-core/dist/types").TensorLike | import("@tensorflow/tfjs-core/dist/tensor").Tensor3D, method?: string, inverted?: boolean, threshValue?: number) => import("@tensorflow/tfjs-core/dist/tensor").Tensor3D;
    transform: (image: import("@tensorflow/tfjs-core/dist/types").TensorLike | import("@tensorflow/tfjs-core/dist/tensor").Tensor4D, transforms: import("@tensorflow/tfjs-core/dist/types").TensorLike | import("@tensorflow/tfjs-core/dist/tensor").Tensor2D, interpolation?: "bilinear" | "nearest", fillMode?: "reflect" | "nearest" | "constant" | "wrap", fillValue?: number, outputShape?: [number, number]) => import("@tensorflow/tfjs-core/dist/tensor").Tensor4D;
};
declare const linalg: {
    bandPart: <T extends import("@tensorflow/tfjs-core/dist/tensor").Tensor<import("@tensorflow/tfjs-core/dist/types").Rank>>(a: import("@tensorflow/tfjs-core/dist/types").TensorLike | T, numLower: number, numUpper: number) => T;
    gramSchmidt: (xs: import("@tensorflow/tfjs-core/dist/tensor").Tensor2D | import("@tensorflow/tfjs-core/dist/tensor").Tensor1D[]) => import("@tensorflow/tfjs-core/dist/tensor").Tensor2D | import("@tensorflow/tfjs-core/dist/tensor").Tensor1D[];
    qr: (x: import("@tensorflow/tfjs-core/dist/tensor").Tensor<import("@tensorflow/tfjs-core/dist/types").Rank>, fullMatrices?: boolean) => [import("@tensorflow/tfjs-core/dist/tensor").Tensor<import("@tensorflow/tfjs-core/dist/types").Rank>, import("@tensorflow/tfjs-core/dist/tensor").Tensor<import("@tensorflow/tfjs-core/dist/types").Rank>];
};
declare const losses: {
    absoluteDifference: <T extends import("@tensorflow/tfjs-core/dist/tensor").Tensor<import("@tensorflow/tfjs-core/dist/types").Rank>, O extends import("@tensorflow/tfjs-core/dist/tensor").Tensor<import("@tensorflow/tfjs-core/dist/types").Rank>>(labels: import("@tensorflow/tfjs-core/dist/types").TensorLike | T, predictions: import("@tensorflow/tfjs-core/dist/types").TensorLike | T, weights?: import("@tensorflow/tfjs-core/dist/types").TensorLike | import("@tensorflow/tfjs-core/dist/tensor").Tensor<import("@tensorflow/tfjs-core/dist/types").Rank>, reduction?: import("@tensorflow/tfjs-core/dist/base").Reduction) => O;
    computeWeightedLoss: <T_1 extends import("@tensorflow/tfjs-core/dist/tensor").Tensor<import("@tensorflow/tfjs-core/dist/types").Rank>, O_1 extends import("@tensorflow/tfjs-core/dist/tensor").Tensor<import("@tensorflow/tfjs-core/dist/types").Rank>>(losses: import("@tensorflow/tfjs-core/dist/types").TensorLike | T_1, weights?: import("@tensorflow/tfjs-core/dist/types").TensorLike | import("@tensorflow/tfjs-core/dist/tensor").Tensor<import("@tensorflow/tfjs-core/dist/types").Rank>, reduction?: import("@tensorflow/tfjs-core/dist/base").Reduction) => O_1;
    cosineDistance: <T_2 extends import("@tensorflow/tfjs-core/dist/tensor").Tensor<import("@tensorflow/tfjs-core/dist/types").Rank>, O_2 extends import("@tensorflow/tfjs-core/dist/tensor").Tensor<import("@tensorflow/tfjs-core/dist/types").Rank>>(labels: import("@tensorflow/tfjs-core/dist/types").TensorLike | T_2, predictions: import("@tensorflow/tfjs-core/dist/types").TensorLike | T_2, axis: number, weights?: import("@tensorflow/tfjs-core/dist/types").TensorLike | import("@tensorflow/tfjs-core/dist/tensor").Tensor<import("@tensorflow/tfjs-core/dist/types").Rank>, reduction?: import("@tensorflow/tfjs-core/dist/base").Reduction) => O_2;
    hingeLoss: <T_3 extends import("@tensorflow/tfjs-core/dist/tensor").Tensor<import("@tensorflow/tfjs-core/dist/types").Rank>, O_3 extends import("@tensorflow/tfjs-core/dist/tensor").Tensor<import("@tensorflow/tfjs-core/dist/types").Rank>>(labels: import("@tensorflow/tfjs-core/dist/types").TensorLike | T_3, predictions: import("@tensorflow/tfjs-core/dist/types").TensorLike | T_3, weights?: import("@tensorflow/tfjs-core/dist/types").TensorLike | import("@tensorflow/tfjs-core/dist/tensor").Tensor<import("@tensorflow/tfjs-core/dist/types").Rank>, reduction?: import("@tensorflow/tfjs-core/dist/base").Reduction) => O_3;
    huberLoss: <T_4 extends import("@tensorflow/tfjs-core/dist/tensor").Tensor<import("@tensorflow/tfjs-core/dist/types").Rank>, O_4 extends import("@tensorflow/tfjs-core/dist/tensor").Tensor<import("@tensorflow/tfjs-core/dist/types").Rank>>(labels: import("@tensorflow/tfjs-core/dist/types").TensorLike | T_4, predictions: import("@tensorflow/tfjs-core/dist/types").TensorLike | T_4, weights?: import("@tensorflow/tfjs-core/dist/types").TensorLike | import("@tensorflow/tfjs-core/dist/tensor").Tensor<import("@tensorflow/tfjs-core/dist/types").Rank>, delta?: number, reduction?: import("@tensorflow/tfjs-core/dist/base").Reduction) => O_4;
    logLoss: <T_5 extends import("@tensorflow/tfjs-core/dist/tensor").Tensor<import("@tensorflow/tfjs-core/dist/types").Rank>, O_5 extends import("@tensorflow/tfjs-core/dist/tensor").Tensor<import("@tensorflow/tfjs-core/dist/types").Rank>>(labels: import("@tensorflow/tfjs-core/dist/types").TensorLike | T_5, predictions: import("@tensorflow/tfjs-core/dist/types").TensorLike | T_5, weights?: import("@tensorflow/tfjs-core/dist/types").TensorLike | import("@tensorflow/tfjs-core/dist/tensor").Tensor<import("@tensorflow/tfjs-core/dist/types").Rank>, epsilon?: number, reduction?: import("@tensorflow/tfjs-core/dist/base").Reduction) => O_5;
    meanSquaredError: <T_6 extends import("@tensorflow/tfjs-core/dist/tensor").Tensor<import("@tensorflow/tfjs-core/dist/types").Rank>, O_6 extends import("@tensorflow/tfjs-core/dist/tensor").Tensor<import("@tensorflow/tfjs-core/dist/types").Rank>>(labels: import("@tensorflow/tfjs-core/dist/types").TensorLike | T_6, predictions: import("@tensorflow/tfjs-core/dist/types").TensorLike | T_6, weights?: import("@tensorflow/tfjs-core/dist/types").TensorLike | import("@tensorflow/tfjs-core/dist/tensor").Tensor<import("@tensorflow/tfjs-core/dist/types").Rank>, reduction?: import("@tensorflow/tfjs-core/dist/base").Reduction) => O_6;
    sigmoidCrossEntropy: <T_7 extends import("@tensorflow/tfjs-core/dist/tensor").Tensor<import("@tensorflow/tfjs-core/dist/types").Rank>, O_7 extends import("@tensorflow/tfjs-core/dist/tensor").Tensor<import("@tensorflow/tfjs-core/dist/types").Rank>>(multiClassLabels: import("@tensorflow/tfjs-core/dist/types").TensorLike | T_7, logits: import("@tensorflow/tfjs-core/dist/types").TensorLike | T_7, weights?: import("@tensorflow/tfjs-core/dist/types").TensorLike | import("@tensorflow/tfjs-core/dist/tensor").Tensor<import("@tensorflow/tfjs-core/dist/types").Rank>, labelSmoothing?: number, reduction?: import("@tensorflow/tfjs-core/dist/base").Reduction) => O_7;
    softmaxCrossEntropy: <T_8 extends import("@tensorflow/tfjs-core/dist/tensor").Tensor<import("@tensorflow/tfjs-core/dist/types").Rank>, O_8 extends import("@tensorflow/tfjs-core/dist/tensor").Tensor<import("@tensorflow/tfjs-core/dist/types").Rank>>(onehotLabels: import("@tensorflow/tfjs-core/dist/types").TensorLike | T_8, logits: import("@tensorflow/tfjs-core/dist/types").TensorLike | T_8, weights?: import("@tensorflow/tfjs-core/dist/types").TensorLike | import("@tensorflow/tfjs-core/dist/tensor").Tensor<import("@tensorflow/tfjs-core/dist/types").Rank>, labelSmoothing?: number, reduction?: import("@tensorflow/tfjs-core/dist/base").Reduction) => O_8;
};
declare const sparse: {
    sparseFillEmptyRows: (indices: import("@tensorflow/tfjs-core/dist/types").TensorLike | import("@tensorflow/tfjs-core/dist/tensor").Tensor2D, values: import("@tensorflow/tfjs-core/dist/types").TensorLike | import("@tensorflow/tfjs-core/dist/tensor").Tensor1D, denseShape: import("@tensorflow/tfjs-core/dist/types").TensorLike | import("@tensorflow/tfjs-core/dist/tensor").Tensor1D, defaultValue: import("@tensorflow/tfjs-core/dist/types").ScalarLike | import("@tensorflow/tfjs-core/dist/tensor").Scalar) => import("@tensorflow/tfjs-core/dist/tensor_types").NamedTensorMap;
    sparseReshape: (inputIndices: import("@tensorflow/tfjs-core/dist/types").TensorLike | import("@tensorflow/tfjs-core/dist/tensor").Tensor2D, inputShape: import("@tensorflow/tfjs-core/dist/types").TensorLike | import("@tensorflow/tfjs-core/dist/tensor").Tensor1D, newShape: import("@tensorflow/tfjs-core/dist/types").TensorLike | import("@tensorflow/tfjs-core/dist/tensor").Tensor1D) => import("@tensorflow/tfjs-core/dist/tensor_types").NamedTensorMap;
    sparseSegmentMean: (data: import("@tensorflow/tfjs-core/dist/types").TensorLike | import("@tensorflow/tfjs-core/dist/tensor").Tensor<import("@tensorflow/tfjs-core/dist/types").Rank>, indices: import("@tensorflow/tfjs-core/dist/types").TensorLike | import("@tensorflow/tfjs-core/dist/tensor").Tensor1D, segmentIds: import("@tensorflow/tfjs-core/dist/types").TensorLike | import("@tensorflow/tfjs-core/dist/tensor").Tensor1D) => import("@tensorflow/tfjs-core/dist/tensor").Tensor<import("@tensorflow/tfjs-core/dist/types").Rank>;
    sparseSegmentSum: (data: import("@tensorflow/tfjs-core/dist/types").TensorLike | import("@tensorflow/tfjs-core/dist/tensor").Tensor<import("@tensorflow/tfjs-core/dist/types").Rank>, indices: import("@tensorflow/tfjs-core/dist/types").TensorLike | import("@tensorflow/tfjs-core/dist/tensor").Tensor1D, segmentIds: import("@tensorflow/tfjs-core/dist/types").TensorLike | import("@tensorflow/tfjs-core/dist/tensor").Tensor1D) => import("@tensorflow/tfjs-core/dist/tensor").Tensor<import("@tensorflow/tfjs-core/dist/types").Rank>;
};
declare const string: {
    stringNGrams: (data: import("@tensorflow/tfjs-core/dist/types").TensorLike | import("@tensorflow/tfjs-core/dist/tensor").Tensor1D, dataSplits: import("@tensorflow/tfjs-core/dist/types").TensorLike | import("@tensorflow/tfjs-core/dist/tensor").Tensor<import("@tensorflow/tfjs-core/dist/types").Rank>, separator: string, nGramWidths: number[], leftPad: string, rightPad: string, padWidth: number, preserveShortSequences: boolean) => import("@tensorflow/tfjs-core/dist/tensor_types").NamedTensorMap;
    stringSplit: (input: import("@tensorflow/tfjs-core/dist/types").TensorLike | import("@tensorflow/tfjs-core/dist/tensor").Tensor1D, delimiter: import("@tensorflow/tfjs-core/dist/types").ScalarLike | import("@tensorflow/tfjs-core/dist/tensor").Scalar, skipEmpty?: boolean) => import("@tensorflow/tfjs-core/dist/tensor_types").NamedTensorMap;
    stringToHashBucketFast: (input: import("@tensorflow/tfjs-core/dist/types").TensorLike | import("@tensorflow/tfjs-core/dist/tensor").Tensor<import("@tensorflow/tfjs-core/dist/types").Rank>, numBuckets: number) => import("@tensorflow/tfjs-core/dist/tensor").Tensor<import("@tensorflow/tfjs-core/dist/types").Rank>;
};
{ image, linalg, losses, spectral, fused, signal, sparse, string };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/ops_for_converter" />
/**
 * This file exports ops used by the converters executors. By default it
 * re-exports all ops. In a custom build this is aliased to a file that will
 * only exports ops for a given model.json.
 */

/// <amd-module name="@tensorflow/tfjs-core/dist/optimizers/optimizer" />
/**
 * A variable that belongs to an optimizer.
 *
 * The `originalName` field is required for keeping track of the canonical
 * name of the variable, which is usually the name of the model weight that
 * the variable is related to plus a suffix, e.g., 'dense1/kernel/momentum'.
 * The name of the `Variable` object itself cannot be used directly due to
 * possible deduplication: Every `Variable` must have a unique name but more
 * than one optimizer objects of the same type may be created for the same model
 * or the same `Variable`.
 */
interface OptimizerVariable {
    originalName: string;
    variable: Variable;
}
/** @doc {heading: 'Training', subheading: 'Classes', namespace: 'train'} */
declare abstract class Optimizer extends Serializable {
    protected iterations_: number;
    /**
     * Executes `f()` and minimizes the scalar output of `f()` by computing
     * gradients of y with respect to the list of trainable variables provided by
     * `varList`. If no list is provided, it defaults to all trainable variables.
     *
     * @param f The function to execute and whose output to minimize.
     * @param returnCost Whether to return the scalar cost value produced by
     * executing `f()`.
     * @param varList An optional list of variables to update. If specified, only
     * the trainable variables in varList will be updated by minimize. Defaults to
     * all trainable variables.
     *
     * @doc {heading: 'Training', subheading: 'Optimizers'}
     */
    minimize(f: () => Scalar, returnCost?: boolean, varList?: Variable[]): Scalar | null;
    /**
     * The number of iterations that this optimizer instance has been invoked for.
     */
    get iterations(): number;
    protected incrementIterations(): void;
    /**
     * Executes f() and computes the gradient of the scalar output of f() with
     * respect to the list of trainable variables provided by `varList`. If no
     * list is provided, it defaults to all trainable variables.
     *
     * @param f The function to execute and whose output to use for computing
     * gradients with respect to variables.
     * @param varList An optional list of variables to compute gradients with
     * respect to. If specified, only the trainable variables in varList will have
     * gradients computed with respect to. Defaults to all trainable variables.
     *
     * @doc {heading: 'Training', subheading: 'Optimizers'}
     */
    computeGradients(f: () => Scalar, varList?: Variable[]): {
        value: Scalar;
        grads: NamedTensorMap;
    };
    /**
     * Updates variables by using the computed gradients.
     *
     * @param variableGradients A mapping of variable name to its gradient value.
     *
     * @doc {heading: 'Training', subheading: 'Optimizers'}
     */
    abstract applyGradients(variableGradients: NamedTensorMap | NamedTensor[]): void;
    /**
     * Dispose the variables (if any) owned by this optimizer instance.
     */
    dispose(): void;
    saveIterations(): Promise<NamedTensor>;
    getWeights(): Promise<NamedTensor[]>;
    setWeights(weightValues: NamedTensor[]): Promise<void>;
    /**
     * Extract the first element of the weight values and set it
     * as the iterations counter variable of this instance of optimizer.
     *
     * @param weightValues
     * @returns Weight values with the first element consumed and excluded.
     */
    protected extractIterations(weightValues: NamedTensor[]): Promise<NamedTensor[]>;
}

/// <amd-module name="@tensorflow/tfjs-layers/dist/optimizers" />
/**
 * Optimizers.
 */
declare function getOptimizer(identifier: string): Optimizer;

/// <amd-module name="@tensorflow/tfjs-layers/dist/keras_format/optimizer_config" />
declare type AdadeltaOptimizerConfig = {
    learning_rate: number;
    rho: number;
    epsilon: number;
};
declare type AdadeltaSerialization = BaseSerialization<'Adadelta', AdadeltaOptimizerConfig>;
declare type AdagradOptimizerConfig = {
    learning_rate: number;
    initial_accumulator_value?: number;
};
declare type AdagradSerialization = BaseSerialization<'Adagrad', AdagradOptimizerConfig>;
declare type AdamOptimizerConfig = {
    learning_rate: number;
    beta1: number;
    beta2: number;
    epsilon?: number;
};
declare type AdamSerialization = BaseSerialization<'Adam', AdamOptimizerConfig>;
declare type AdamaxOptimizerConfig = {
    learning_rate: number;
    beta1: number;
    beta2: number;
    epsilon?: number;
    decay?: number;
};
declare type AdamaxSerialization = BaseSerialization<'Adamax', AdamaxOptimizerConfig>;
declare type MomentumOptimizerConfig = {
    learning_rate: number;
    momentum: number;
    use_nesterov?: boolean;
};
declare type MomentumSerialization = BaseSerialization<'Momentum', MomentumOptimizerConfig>;
declare type RMSPropOptimizerConfig = {
    learning_rate: number;
    decay?: number;
    momentum?: number;
    epsilon?: number;
    centered?: boolean;
};
declare type RMSPropSerialization = BaseSerialization<'RMSProp', RMSPropOptimizerConfig>;
declare type SGDOptimizerConfig = {
    learning_rate: number;
};
declare type SGDSerialization = BaseSerialization<'SGD', SGDOptimizerConfig>;
declare type OptimizerSerialization = AdadeltaSerialization | AdagradSerialization | AdamSerialization | AdamaxSerialization | MomentumSerialization | RMSPropSerialization | SGDSerialization;
declare type OptimizerClassName = OptimizerSerialization['class_name'];
/**
 * A string array of valid Optimizer class names.
 *
 * This is guaranteed to match the `OptimizerClassName` union type.
 */
declare const optimizerClassNames: OptimizerClassName[];

/// <amd-module name="@tensorflow/tfjs-core/dist/optimizers/optimizer_constructors" />
declare class OptimizerConstructors {
    /**
     * Constructs a `tf.SGDOptimizer` that uses stochastic gradient descent.
     *
     * ```js
     * // Fit a quadratic function by learning the coefficients a, b, c.
     * const xs = tf.tensor1d([0, 1, 2, 3]);
     * const ys = tf.tensor1d([1.1, 5.9, 16.8, 33.9]);
     *
     * const a = tf.scalar(Math.random()).variable();
     * const b = tf.scalar(Math.random()).variable();
     * const c = tf.scalar(Math.random()).variable();
     *
     * // y = a * x^2 + b * x + c.
     * const f = x => a.mul(x.square()).add(b.mul(x)).add(c);
     * const loss = (pred, label) => pred.sub(label).square().mean();
     *
     * const learningRate = 0.01;
     * const optimizer = tf.train.sgd(learningRate);
     *
     * // Train the model.
     * for (let i = 0; i < 10; i++) {
     *   optimizer.minimize(() => loss(f(xs), ys));
     * }
     *
     * // Make predictions.
     * console.log(
     *     `a: ${a.dataSync()}, b: ${b.dataSync()}, c: ${c.dataSync()}`);
     * const preds = f(xs).dataSync();
     * preds.forEach((pred, i) => {
     *   console.log(`x: ${i}, pred: ${pred}`);
     * });
     * ```
     *
     * @param learningRate The learning rate to use for the SGD algorithm.
     *
     * @doc {heading: 'Training', subheading: 'Optimizers', namespace: 'train'}
     */
    static sgd(learningRate: number): SGDOptimizer;
    /**
     * Constructs a `tf.MomentumOptimizer` that uses momentum gradient
     * descent.
     *
     * See
     * [http://proceedings.mlr.press/v28/sutskever13.pdf](
     * http://proceedings.mlr.press/v28/sutskever13.pdf)
     *
     * @param learningRate The learning rate to use for the Momentum gradient
     * descent algorithm.
     * @param momentum The momentum to use for the momentum gradient descent
     * algorithm.
     *
     * @doc {heading: 'Training', subheading: 'Optimizers', namespace: 'train'}
     */
    static momentum(learningRate: number, momentum: number, useNesterov?: boolean): MomentumOptimizer;
    /**
     * Constructs a `tf.RMSPropOptimizer` that uses RMSProp gradient
     * descent. This implementation uses plain momentum and is not centered
     * version of RMSProp.
     *
     * See
     * [http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf](
     * http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)
     *
     * @param learningRate The learning rate to use for the RMSProp gradient
     * descent algorithm.
     * @param decay The discounting factor for the history/coming gradient.
     * @param momentum The momentum to use for the RMSProp gradient descent
     * algorithm.
     * @param epsilon Small value to avoid zero denominator.
     * @param centered If true, gradients are normalized by the estimated
     * variance of the gradient.
     *
     * @doc {heading: 'Training', subheading: 'Optimizers', namespace: 'train'}
     */
    static rmsprop(learningRate: number, decay?: number, momentum?: number, epsilon?: number, centered?: boolean): RMSPropOptimizer;
    /**
     * Constructs a `tf.AdamOptimizer` that uses the Adam algorithm.
     * See [https://arxiv.org/abs/1412.6980](https://arxiv.org/abs/1412.6980)
     *
     * @param learningRate The learning rate to use for the Adam gradient
     * descent algorithm.
     * @param beta1 The exponential decay rate for the 1st moment estimates.
     * @param beta2 The exponential decay rate for the 2nd moment estimates.
     * @param epsilon A small constant for numerical stability.
     *
     * @doc {heading: 'Training', subheading: 'Optimizers', namespace: 'train'}
     */
    static adam(learningRate?: number, beta1?: number, beta2?: number, epsilon?: number): AdamOptimizer;
    /**
     * Constructs a `tf.AdadeltaOptimizer` that uses the Adadelta algorithm.
     * See [https://arxiv.org/abs/1212.5701](https://arxiv.org/abs/1212.5701)
     *
     * @param learningRate The learning rate to use for the Adadelta gradient
     * descent algorithm.
     * @param rho The learning rate decay over each update.
     * @param epsilon A constant epsilon used to better condition the grad
     * update.
     *
     * @doc {heading: 'Training', subheading: 'Optimizers', namespace: 'train'}
     */
    static adadelta(learningRate?: number, rho?: number, epsilon?: number): AdadeltaOptimizer;
    /**
     * Constructs a `tf.AdamaxOptimizer` that uses the Adamax algorithm.
     * See [https://arxiv.org/abs/1412.6980](https://arxiv.org/abs/1412.6980)
     *
     * @param learningRate The learning rate to use for the Adamax gradient
     * descent algorithm.
     * @param beta1 The exponential decay rate for the 1st moment estimates.
     * @param beta2 The exponential decay rate for the 2nd moment estimates.
     * @param epsilon A small constant for numerical stability.
     * @param decay The learning rate decay over each update.
     *
     * @doc {heading: 'Training', subheading: 'Optimizers', namespace: 'train'}
     */
    static adamax(learningRate?: number, beta1?: number, beta2?: number, epsilon?: number, decay?: number): AdamaxOptimizer;
    /**
     * Constructs a `tf.AdagradOptimizer` that uses the Adagrad algorithm.
     * See
     * [http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf](
     * http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf)
     * or
     * [http://ruder.io/optimizing-gradient-descent/index.html#adagrad](
     * http://ruder.io/optimizing-gradient-descent/index.html#adagrad)
     *
     * @param learningRate The learning rate to use for the Adagrad gradient
     * descent algorithm.
     * @param initialAccumulatorValue Starting value for the accumulators, must be
     * positive.
     *
     * @doc {heading: 'Training', subheading: 'Optimizers', namespace: 'train'}
     */
    static adagrad(learningRate: number, initialAccumulatorValue?: number): AdagradOptimizer;
}

/// <amd-module name="@tensorflow/tfjs-core/dist/optimizers/optimizer_test" />
{ };
/// <amd-module name="@tensorflow/tfjs-core/dist/ops/outer_product" />

/**
 * Computes the outer product of two vectors, `v1` and `v2`.
 *
 * ```js
 * const a = tf.tensor1d([1, 2, 3]);
 * const b = tf.tensor1d([3, 4, 5]);
 *
 * tf.outerProduct(a, b).print();
 * ```
 * @param v1 The first vector in the outer product operation.
 * @param v2 The second vector in the outer product operation.
 *
 * @doc {heading: 'Operations', subheading: 'Matrices'}
 */
declare function outerProduct_(v1: Tensor1D | TensorLike, v2: Tensor1D | TensorLike): Tensor2D;
declare const outerProduct: typeof outerProduct_;
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/gradients/Pack_grad" />
declare const packGradConfig: GradConfig;

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/pad" />
/**
 * Pads a `tf.Tensor` with a given value and paddings.
 *
 * This operation implements `CONSTANT` mode. For `REFLECT` and `SYMMETRIC`,
 * refer to `tf.mirrorPad`.
 *
 * Also available are stricter rank-specific methods with the same signature
 * as this method that assert that `paddings` is of given length.
 *   - `tf.pad1d`
 *   - `tf.pad2d`
 *   - `tf.pad3d`
 *   - `tf.pad4d`
 *
 * ```js
 * const x = tf.tensor1d([1, 2, 3, 4]);
 * x.pad([[1, 2]]).print();
 * ```
 * @param x The tensor to pad.
 * @param paddings An array of length `R` (the rank of the tensor), where
 * each element is a length-2 tuple of ints `[padBefore, padAfter]`,
 * specifying how much to pad along each dimension of the tensor.
 * @param constantValue The pad value to use. Defaults to 0.
 *
 * @doc {heading: 'Tensors', subheading: 'Transformations'}
 */
declare function pad_<T extends Tensor>(x: T | TensorLike, paddings: Array<[number, number]>, constantValue?: number): T;
declare const pad: typeof pad_;
{ };
/// <amd-module name="@tensorflow/tfjs-core/dist/ops/pad1d" />

/**
 * Pads a `tf.Tensor1D` with a given value and paddings. See `pad` for details.
 */
declare function pad1d_(x: Tensor1D | TensorLike, paddings: [number, number], constantValue?: number): Tensor1D;
declare const pad1d: typeof pad1d_;
{ };
/// <amd-module name="@tensorflow/tfjs-core/dist/ops/pad2d" />

/**
 * Pads a `tf.Tensor2D` with a given value and paddings. See `pad` for details.
 */
declare function pad2d_(x: Tensor2D | TensorLike, paddings: [[number, number], [number, number]], constantValue?: number): Tensor2D;
declare const pad2d: typeof pad2d_;
{ };
/// <amd-module name="@tensorflow/tfjs-core/dist/ops/pad3d" />

/**
 * Pads a `tf.Tensor3D` with a given value and paddings. See `pad` for details.
 */
declare function pad3d_(x: Tensor3D | TensorLike, paddings: [[number, number], [number, number], [number, number]], constantValue?: number): Tensor3D;
declare const pad3d: typeof pad3d_;
{ };
/// <amd-module name="@tensorflow/tfjs-core/dist/ops/pad4d" />

/**
 * Pads a `tf.Tensor4D` with a given value and paddings. See `pad` for details.
 */
declare function pad4d_(x: Tensor4D | TensorLike, paddings: [
    [
        number,
        number
    ],
    [number, number],
    [number, number],
    [number, number]
], constantValue?: number): Tensor4D;
declare const pad4d: typeof pad4d_;
{ };

/// <amd-module name="@tensorflow/tfjs-layers/dist/layers/padding" />
/**
 * Pads the middle dimension of a 3D tensor.
 *
 * @param x Input `tf.Tensor` to be padded.
 * @param padding `Array` of 2 integers, how many zeros to add at the start and
 *   end of the middle dimension (i.e., dimension 1).
 * @return A padded 3D `tf.Tensor`.
 */
declare function temporalPadding(x: Tensor, padding?: [number, number]): Tensor;
/**
 * Pads the 2nd and 3rd dimensions of a 4D tensor.
 *
 * @param x Input `tf.Tensor` to be padded.
 * @param padding `Array` of two `Array`s, each of which is an `Array` of two
 *   integers. The amount of padding at the beginning and end of the 2nd and 3rd
 *   dimensions, respectively.
 * @param dataFormat 'channelsLast' (default) or 'channelsFirst'.
 * @return Padded 4D `tf.Tensor`.
 */
declare function spatial2dPadding(x: Tensor, padding?: [[number, number], [number, number]], dataFormat?: DataFormat): Tensor;
declare interface ZeroPadding2DLayerArgs extends LayerArgs {
    /**
     * Integer, or `Array` of 2 integers, or `Array` of 2 `Array`s, each of
     * which is an `Array` of 2 integers.
     * - If integer, the same symmetric padding is applied to width and height.
     * - If `Array` of 2 integers, interpreted as two different symmetric values
     *   for height and width:
     *   `[symmetricHeightPad, symmetricWidthPad]`.
     * - If `Array` of 2 `Array`s, interpreted as:
     *   `[[topPad, bottomPad], [leftPad, rightPad]]`.
     */
    padding?: number | [number, number] | [[number, number], [number, number]];
    /**
     * One of `'channelsLast'` (default) and `'channelsFirst'`.
     *
     * The ordering of the dimensions in the inputs.
     * `channelsLast` corresponds to inputs with shape
     * `[batch, height, width, channels]` while `channelsFirst`
     * corresponds to inputs with shape
     * `[batch, channels, height, width]`.
     */
    dataFormat?: DataFormat;
}
declare class ZeroPadding2D extends Layer {
    /** @nocollapse */
    static className: string;
    readonly dataFormat: DataFormat;
    readonly padding: [[number, number], [number, number]];
    constructor(args?: ZeroPadding2DLayerArgs);
    computeOutputShape(inputShape: Shape | Shape[]): Shape | Shape[];
    call(inputs: Tensor | Tensor[], kwargs: Kwargs): Tensor | Tensor[];
    getConfig(): serialization.ConfigDict;
}

/// <amd-module name="@tensorflow/tfjs-layers/dist/keras_format/layers/padding_serialization" />
interface ZeroPadding2DLayerConfig extends LayerConfig {
    padding?: number | [number, number] | [[number, number], [number, number]];
    data_format?: DataFormatSerialization;
}
declare type ZeroPadding2DLayerSerialization = BaseLayerSerialization<'ZeroPadding2D', ZeroPadding2DLayerConfig>;
declare type PaddingLayerSerialization = ZeroPadding2DLayerSerialization;
declare type PaddingLayerClassName = PaddingLayerSerialization['class_name'];
/**
 * A string array of valid PaddingLayer class names.
 *
 * This is guaranteed to match the `PaddingLayerClassName` union type.
 */
declare const paddingLayerClassNames: PaddingLayerClassName[];

/// <amd-module name="@tensorflow/tfjs-core/dist/gradients/PadV2_grad" />
declare const padV2GradConfig: GradConfig;

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/pad_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/io/passthrough" />
/**
 * IOHandlers that pass through the in-memory ModelArtifacts format.
 */
/**
 * Creates an IOHandler that loads model artifacts from memory.
 *
 * When used in conjunction with `tf.loadLayersModel`, an instance of
 * `tf.LayersModel` (Keras-style) can be constructed from the loaded artifacts.
 *
 * ```js
 * const model = await tf.loadLayersModel(tf.io.fromMemory(
 *     modelTopology, weightSpecs, weightData));
 * ```
 *
 * @param modelArtifacts a object containing model topology (i.e., parsed from
 *   the JSON format).
 * @param weightSpecs An array of `WeightsManifestEntry` objects describing the
 *   names, shapes, types, and quantization of the weight data. Optional.
 * @param weightData A single `ArrayBuffer` containing the weight data,
 *   concatenated in the order described by the weightSpecs. Optional.
 * @param trainingConfig Model training configuration. Optional.
 *
 * @returns A passthrough `IOHandler` that simply loads the provided data.
 */
declare function fromMemory(modelArtifacts: {} | ModelArtifacts, weightSpecs?: WeightsManifestEntry[], weightData?: ArrayBuffer, trainingConfig?: TrainingConfig): IOHandler;
/**
 * Creates an IOHandler that loads model artifacts from memory.
 *
 * When used in conjunction with `tf.loadLayersModel`, an instance of
 * `tf.LayersModel` (Keras-style) can be constructed from the loaded artifacts.
 *
 * ```js
 * const model = await tf.loadLayersModel(tf.io.fromMemory(
 *     modelTopology, weightSpecs, weightData));
 * ```
 *
 * @param modelArtifacts a object containing model topology (i.e., parsed from
 *   the JSON format).
 * @param weightSpecs An array of `WeightsManifestEntry` objects describing the
 *   names, shapes, types, and quantization of the weight data. Optional.
 * @param weightData A single `ArrayBuffer` containing the weight data,
 *   concatenated in the order described by the weightSpecs. Optional.
 * @param trainingConfig Model training configuration. Optional.
 *
 * @returns A passthrough `IOHandlerSync` that simply loads the provided data.
 */
declare function fromMemorySync(modelArtifacts: {} | ModelArtifacts, weightSpecs?: WeightsManifestEntry[], weightData?: ArrayBuffer, trainingConfig?: TrainingConfig): IOHandlerSync;
/**
 * Creates an IOHandler that passes saved model artifacts to a callback.
 *
 * ```js
 * function handleSave(artifacts) {
 *   // ... do something with the artifacts ...
 *   return {modelArtifactsInfo: {...}, ...};
 * }
 *
 * const saveResult = model.save(tf.io.withSaveHandler(handleSave));
 * ```
 *
 * @param saveHandler A function that accepts a `ModelArtifacts` and returns a
 *     promise that resolves to a `SaveResult`.
 */
declare function withSaveHandler(saveHandler: (artifacts: ModelArtifacts) => Promise<SaveResult>): IOHandler;
/**
 * Creates an IOHandlerSync that passes saved model artifacts to a callback.
 *
 * ```js
 * function handleSave(artifacts) {
 *   // ... do something with the artifacts ...
 *   return {modelArtifactsInfo: {...}, ...};
 * }
 *
 * const saveResult = model.save(tf.io.withSaveHandler(handleSave));
 * ```
 *
 * @param saveHandler A function that accepts a `ModelArtifacts` and returns a
 *     `SaveResult`.
 */
declare function withSaveHandlerSync(saveHandler: (artifacts: ModelArtifacts) => SaveResult): IOHandlerSync;

/// <amd-module name="@tensorflow/tfjs-core/dist/io/passthrough_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/platforms/platform" />
/**
 * At any given time a single platform is active and represents and
 * implementation of this interface. In practice, a platform is an environment
 * where TensorFlow.js can be executed, e.g. the browser or Node.js.
 */
interface Platform {
    /**
     * Makes an HTTP request.
     * @param path The URL path to make a request to
     * @param init The request init. See init here:
     *     https://developer.mozilla.org/en-US/docs/Web/API/Request/Request
     */
    fetch(path: string, requestInits?: RequestInit, options?: RequestDetails): Promise<Response>;
    /**
     * Returns the current high-resolution time in milliseconds relative to an
     * arbitrary time in the past. It works across different platforms (node.js,
     * browsers).
     */
    now(): number;
    /**
     * Encode the provided string into an array of bytes using the provided
     * encoding.
     */
    encode(text: string, encoding: string): Uint8Array;
    /** Decode the provided bytes into a string using the provided encoding. */
    decode(bytes: Uint8Array, encoding: string): string;
    setTimeoutCustom?(functionRef: Function, delay: number): void;
    isTypedArray(a: unknown): a is Float32Array | Int32Array | Uint8Array | Uint8ClampedArray;
}

/// <amd-module name="@tensorflow/tfjs-core/dist/platforms/platform_browser" />
declare class PlatformBrowser implements Platform {
    private textEncoder;
    private readonly messageName;
    private functionRefs;
    private handledMessageCount;
    private hasEventListener;
    fetch(path: string, init?: RequestInit): Promise<Response>;
    now(): number;
    encode(text: string, encoding: string): Uint8Array;
    decode(bytes: Uint8Array, encoding: string): string;
    setTimeoutCustom(functionRef: Function, delay: number): void;
    isTypedArray(a: unknown): a is Uint8Array | Float32Array | Int32Array | Uint8ClampedArray;
}

/// <amd-module name="@tensorflow/tfjs-core/dist/platforms/platform_browser_test" />
{ };
/// <amd-module name="@tensorflow/tfjs-core/dist/platforms/platform_node" />
declare const getNodeFetch: {
    importFetch: () => any;
};
declare type FetchFn = (url: string, init?: RequestInit) => Promise<Response>;
declare function resetSystemFetch(): void;
declare function setSystemFetch(fetchFn: FetchFn): void;
declare function getSystemFetch(): FetchFn;
declare class PlatformNode implements Platform {
    private textEncoder;
    util: any;
    constructor();
    fetch(path: string, requestInits?: RequestInit): Promise<Response>;
    now(): number;
    encode(text: string, encoding: string): Uint8Array;
    decode(bytes: Uint8Array, encoding: string): string;
    isTypedArray(a: unknown): a is Float32Array | Int32Array | Uint8Array | Uint8ClampedArray;
}
{ };
/// <amd-module name="@tensorflow/tfjs-core/dist/public/chained_ops/pool" />

declare module '../../tensor' {
    interface Tensor<R extends Rank = Rank> {
        pool<T extends Tensor3D | Tensor4D>(windowShape: [number, number] | number, poolingType: 'avg' | 'max', padding: 'valid' | 'same' | number | ExplicitPadding, diationRate?: [number, number] | number, strides?: [number, number] | number, dimRoundingMode?: 'floor' | 'round' | 'ceil'): T;
    }
}

/// <amd-module name="@tensorflow/tfjs-layers/dist/layers/pooling" />
/**
 * 2D pooling.
 * @param x
 * @param poolSize
 * @param strides strides. Defaults to [1, 1].
 * @param padding padding. Defaults to 'valid'.
 * @param dataFormat data format. Defaults to 'channelsLast'.
 * @param poolMode Mode of pooling. Defaults to 'max'.
 * @returns Result of the 2D pooling.
 */
declare function pool2d(x: Tensor, poolSize: [number, number], strides?: [number, number], padding?: PaddingMode, dataFormat?: DataFormat, poolMode?: PoolMode): Tensor;
/**
 * 3D pooling.
 * @param x
 * @param poolSize. Default to [1, 1, 1].
 * @param strides strides. Defaults to [1, 1, 1].
 * @param padding padding. Defaults to 'valid'.
 * @param dataFormat data format. Defaults to 'channelsLast'.
 * @param poolMode Mode of pooling. Defaults to 'max'.
 * @returns Result of the 3D pooling.
 */
declare function pool3d(x: Tensor5D, poolSize: [number, number, number], strides?: [number, number, number], padding?: PaddingMode, dataFormat?: DataFormat, poolMode?: PoolMode): Tensor;
declare interface Pooling1DLayerArgs extends LayerArgs {
    /**
     * Size of the window to pool over, should be an integer.
     */
    poolSize?: number | [number];
    /**
     * Period at which to sample the pooled values.
     *
     * If `null`, defaults to `poolSize`.
     */
    strides?: number | [number];
    /** How to fill in data that's not an integer multiple of poolSize. */
    padding?: PaddingMode;
}
/**
 * Abstract class for different pooling 1D layers.
 */
declare abstract class Pooling1D extends Layer {
    protected readonly poolSize: [number];
    protected readonly strides: [number];
    protected readonly padding: PaddingMode;
    /**
     *
     * @param args Parameters for the Pooling layer.
     *
     * config.poolSize defaults to 2.
     */
    constructor(args: Pooling1DLayerArgs);
    computeOutputShape(inputShape: Shape | Shape[]): Shape | Shape[];
    protected abstract poolingFunction(inputs: Tensor, poolSize: [number, number], strides: [number, number], padding: PaddingMode, dataFormat: DataFormat): Tensor;
    call(inputs: Tensor | Tensor[], kwargs: Kwargs): Tensor | Tensor[];
    getConfig(): serialization.ConfigDict;
}
declare class MaxPooling1D extends Pooling1D {
    /** @nocollapse */
    static className: string;
    constructor(args: Pooling1DLayerArgs);
    protected poolingFunction(inputs: Tensor, poolSize: [number, number], strides: [number, number], padding: PaddingMode, dataFormat: DataFormat): Tensor;
}
declare class AveragePooling1D extends Pooling1D {
    /** @nocollapse */
    static className: string;
    constructor(args: Pooling1DLayerArgs);
    protected poolingFunction(inputs: Tensor, poolSize: [number, number], strides: [number, number], padding: PaddingMode, dataFormat: DataFormat): Tensor;
}
declare interface Pooling2DLayerArgs extends LayerArgs {
    /**
     * Factors by which to downscale in each dimension [vertical, horizontal].
     * Expects an integer or an array of 2 integers.
     *
     * For example, `[2, 2]` will halve the input in both spatial dimensions.
     * If only one integer is specified, the same window length
     * will be used for both dimensions.
     */
    poolSize?: number | [number, number];
    /**
     * The size of the stride in each dimension of the pooling window. Expects
     * an integer or an array of 2 integers. Integer, tuple of 2 integers, or
     * None.
     *
     * If `null`, defaults to `poolSize`.
     */
    strides?: number | [number, number];
    /** The padding type to use for the pooling layer. */
    padding?: PaddingMode;
    /** The data format to use for the pooling layer. */
    dataFormat?: DataFormat;
}
/**
 * Abstract class for different pooling 2D layers.
 */
declare abstract class Pooling2D extends Layer {
    protected readonly poolSize: [number, number];
    protected readonly strides: [number, number];
    protected readonly padding: PaddingMode;
    protected readonly dataFormat: DataFormat;
    constructor(args: Pooling2DLayerArgs);
    computeOutputShape(inputShape: Shape | Shape[]): Shape | Shape[];
    protected abstract poolingFunction(inputs: Tensor, poolSize: [number, number], strides: [number, number], padding: PaddingMode, dataFormat: DataFormat): Tensor;
    call(inputs: Tensor | Tensor[], kwargs: Kwargs): Tensor | Tensor[];
    getConfig(): serialization.ConfigDict;
}
declare class MaxPooling2D extends Pooling2D {
    /** @nocollapse */
    static className: string;
    constructor(args: Pooling2DLayerArgs);
    protected poolingFunction(inputs: Tensor, poolSize: [number, number], strides: [number, number], padding: PaddingMode, dataFormat: DataFormat): Tensor;
}
declare class AveragePooling2D extends Pooling2D {
    /** @nocollapse */
    static className: string;
    constructor(args: Pooling2DLayerArgs);
    protected poolingFunction(inputs: Tensor, poolSize: [number, number], strides: [number, number], padding: PaddingMode, dataFormat: DataFormat): Tensor;
}
declare interface Pooling3DLayerArgs extends LayerArgs {
    /**
     * Factors by which to downscale in each dimension [depth, height, width].
     * Expects an integer or an array of 3 integers.
     *
     * For example, `[2, 2, 2]` will halve the input in three dimensions.
     * If only one integer is specified, the same window length
     * will be used for all dimensions.
     */
    poolSize?: number | [number, number, number];
    /**
     * The size of the stride in each dimension of the pooling window. Expects
     * an integer or an array of 3 integers. Integer, tuple of 3 integers, or
     * None.
     *
     * If `null`, defaults to `poolSize`.
     */
    strides?: number | [number, number, number];
    /** The padding type to use for the pooling layer. */
    padding?: PaddingMode;
    /** The data format to use for the pooling layer. */
    dataFormat?: DataFormat;
}
/**
 * Abstract class for different pooling 3D layers.
 */
declare abstract class Pooling3D extends Layer {
    protected readonly poolSize: [number, number, number];
    protected readonly strides: [number, number, number];
    protected readonly padding: PaddingMode;
    protected readonly dataFormat: DataFormat;
    constructor(args: Pooling3DLayerArgs);
    computeOutputShape(inputShape: Shape | Shape[]): Shape | Shape[];
    protected abstract poolingFunction(inputs: Tensor, poolSize: [number, number, number], strides: [number, number, number], padding: PaddingMode, dataFormat: DataFormat): Tensor;
    call(inputs: Tensor | Tensor[], kwargs: Kwargs): Tensor | Tensor[];
    getConfig(): serialization.ConfigDict;
}
declare class MaxPooling3D extends Pooling3D {
    /** @nocollapse */
    static className: string;
    constructor(args: Pooling3DLayerArgs);
    protected poolingFunction(inputs: Tensor, poolSize: [number, number, number], strides: [number, number, number], padding: PaddingMode, dataFormat: DataFormat): Tensor;
}
declare class AveragePooling3D extends Pooling3D {
    /** @nocollapse */
    static className: string;
    constructor(args: Pooling3DLayerArgs);
    protected poolingFunction(inputs: Tensor, poolSize: [number, number, number], strides: [number, number, number], padding: PaddingMode, dataFormat: DataFormat): Tensor;
}
/**
 * Abstract class for different global pooling 1D layers.
 */
declare abstract class GlobalPooling1D extends Layer {
    constructor(args: LayerArgs);
    computeOutputShape(inputShape: Shape): Shape;
    call(inputs: Tensor | Tensor[], kwargs: Kwargs): Tensor | Tensor[];
}
declare class GlobalAveragePooling1D extends GlobalPooling1D {
    /** @nocollapse */
    static className: string;
    constructor(args?: LayerArgs);
    call(inputs: Tensor | Tensor[], kwargs: Kwargs): Tensor | Tensor[];
}
declare class GlobalMaxPooling1D extends GlobalPooling1D {
    /** @nocollapse */
    static className: string;
    constructor(args: LayerArgs);
    call(inputs: Tensor | Tensor[], kwargs: Kwargs): Tensor | Tensor[];
}
declare interface GlobalPooling2DLayerArgs extends LayerArgs {
    /**
     * One of `CHANNEL_LAST` (default) or `CHANNEL_FIRST`.
     *
     * The ordering of the dimensions in the inputs. `CHANNEL_LAST` corresponds
     * to inputs with shape `[batch, height, width, channels]` while
     * `CHANNEL_FIRST` corresponds to inputs with shape
     * `[batch, channels, height, width]`.
     */
    dataFormat?: DataFormat;
}
/**
 * Abstract class for different global pooling 2D layers.
 */
declare abstract class GlobalPooling2D extends Layer {
    protected dataFormat: DataFormat;
    constructor(args: GlobalPooling2DLayerArgs);
    computeOutputShape(inputShape: Shape | Shape[]): Shape | Shape[];
    call(inputs: Tensor | Tensor[], kwargs: Kwargs): Tensor | Tensor[];
    getConfig(): serialization.ConfigDict;
}
declare class GlobalAveragePooling2D extends GlobalPooling2D {
    /** @nocollapse */
    static className: string;
    call(inputs: Tensor | Tensor[], kwargs: Kwargs): Tensor | Tensor[];
}
declare class GlobalMaxPooling2D extends GlobalPooling2D {
    /** @nocollapse */
    static className: string;
    call(inputs: Tensor | Tensor[], kwargs: Kwargs): Tensor | Tensor[];
}

/// <amd-module name="@tensorflow/tfjs-layers/dist/keras_format/layers/pooling_serialization" />
interface Pooling1DLayerConfig extends LayerConfig {
    pool_size?: [number];
    strides?: [number];
    padding?: PaddingMode;
}
declare type MaxPooling1DLayerSerialization = BaseLayerSerialization<'MaxPooling1D', Pooling1DLayerConfig>;
declare type AveragePooling1DLayerSerialization = BaseLayerSerialization<'AveragePooling1D', Pooling1DLayerConfig>;
interface Pooling2DLayerConfig extends LayerConfig {
    pool_size?: number | [number, number];
    strides?: number | [number, number];
    padding?: PaddingMode;
    data_format?: DataFormatSerialization;
}
declare type MaxPooling2DLayerSerialization = BaseLayerSerialization<'MaxPooling2D', Pooling2DLayerConfig>;
declare type AveragePooling2DLayerSerialization = BaseLayerSerialization<'AveragePooling2D', Pooling2DLayerConfig>;
declare type GlobalAveragePooling1DLayerSerialization = BaseLayerSerialization<'GlobalAveragePooling1D', LayerConfig>;
declare type GlobalMaxPooling1DLayerSerialization = BaseLayerSerialization<'GlobalMaxPooling1D', LayerConfig>;
interface GlobalPooling2DLayerConfig extends LayerConfig {
    data_format?: DataFormatSerialization;
}
declare type GlobalAveragePooling2DLayerSerialization = BaseLayerSerialization<'GlobalAveragePooling2D', GlobalPooling2DLayerConfig>;
declare type GlobalMaxPooling2DLayerSerialization = BaseLayerSerialization<'GlobalMaxPooling2D', GlobalPooling2DLayerConfig>;
declare type PoolingLayerSerialization = MaxPooling1DLayerSerialization | AveragePooling1DLayerSerialization | MaxPooling2DLayerSerialization | AveragePooling2DLayerSerialization | GlobalAveragePooling1DLayerSerialization | GlobalMaxPooling1DLayerSerialization | GlobalAveragePooling2DLayerSerialization | GlobalMaxPooling2DLayerSerialization;
declare type PoolingLayerClassName = PoolingLayerSerialization['class_name'];
/**
 * A string array of valid PoolingLayer class names.
 *
 * This is guaranteed to match the `PoolingLayerClassName` union type.
 */
declare const poolingLayerClassNames: PoolingLayerClassName[];

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/pool_test" />
{ };
/// <amd-module name="@tensorflow/tfjs-core/dist/ops/pow" />
/**
 * Computes the power of one `tf.Tensor` to another. Supports broadcasting.
 *
 * Given a `tf.Tensor` x and a `tf.Tensor` y, this operation computes x^y for
 * corresponding elements in x and y. The result's dtype will be the upcasted
 * type of the `base` and `exp` dtypes.
 *
 * ```js
 * const a = tf.tensor([[2, 3], [4, 5]])
 * const b = tf.tensor([[1, 2], [3, 0]]).toInt();
 *
 * a.pow(b).print();  // or tf.pow(a, b)
 * ```
 *
 * ```js
 * const a = tf.tensor([[1, 2], [3, 4]])
 * const b = tf.tensor(2).toInt();
 *
 * a.pow(b).print();  // or tf.pow(a, b)
 * ```
 * We also expose `powStrict` which has the same signature as this op and
 * asserts that `base` and `exp` are the same shape (does not broadcast).
 *
 * @param base The base `tf.Tensor` to pow element-wise.
 * @param exp The exponent `tf.Tensor` to pow element-wise.
 *
 * @doc {heading: 'Operations', subheading: 'Arithmetic'}
 */
declare function pow_<T extends Tensor>(base: Tensor | TensorLike, exp: Tensor | TensorLike): T;
declare const pow: typeof pow_;
{ };
/// <amd-module name="@tensorflow/tfjs-core/dist/gradients/Pow_grad" />
declare const powGradConfig: GradConfig;

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/prelu" />
/**
 * Computes leaky rectified linear element-wise with parametric alphas.
 *
 * `x < 0 ? alpha * x : f(x) = x`
 *
 * ```js
 * const x = tf.tensor1d([-1, 2, -3, 4]);
 * const alpha = tf.scalar(0.1);
 *
 * x.prelu(alpha).print();  // or tf.prelu(x, alpha)
 * ```
 * @param x The input tensor.
 * @param alpha Scaling factor for negative values.
 *
 * @doc {heading: 'Operations', subheading: 'Basic math'}
 */
declare function prelu_<T extends Tensor>(x: T | TensorLike, alpha: T | TensorLike): T;
declare const prelu: typeof prelu_;
{ };
/// <amd-module name="@tensorflow/tfjs-core/dist/gradients/Prelu_grad" />
declare const preluGradConfig: GradConfig;

/// <amd-module name="@tensorflow/tfjs-layers/dist/layers/preprocessing/preprocessing_utils" />
declare type OutputMode = 'int' | 'oneHot' | 'multiHot' | 'count' | 'tfIdf';
declare function encodeCategoricalInputs(inputs: Tensor | Tensor[], outputMode: OutputMode, depth: number, weights?: Tensor1D | Tensor2D | TensorLike): Tensor | Tensor[];

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/print" />
/**
 * Prints information about the `tf.Tensor` including its data.
 *
 * ```js
 * const verbose = true;
 * tf.tensor2d([1, 2, 3, 4], [2, 2]).print(verbose);
 * ```
 * @param x The tensor to be printed.
 * @param verbose Whether to print verbose information about the ` Tensor`,
 * including dtype and size.
 *
 * @doc {heading: 'Tensors', subheading: 'Creation'}
 */
declare function print<T extends Tensor>(x: T, verbose?: boolean): void;
/// <amd-module name="@tensorflow/tfjs-core/dist/public/chained_ops/prod" />
declare module '../../tensor' {
    interface Tensor<R extends Rank = Rank> {
        prod<T extends Tensor>(this: T, axis?: number | number[], keepDims?: boolean): T;
    }
}

/// <amd-module name="@tensorflow/tfjs-core/dist/gradients/Prod_grad" />
declare const prodGradConfig: GradConfig;

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/prod_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/profiler" />
declare type KernelProfile = {
    kernelName: string;
    outputs: Tensor[];
    inputs: NamedTensorMap;
    timeMs: Promise<number | {
        error: string;
    }>;
    extraInfo: Promise<string>;
};
declare class Profiler {
    private backendTimer;
    private logger?;
    constructor(backendTimer: BackendTimer, logger?: Logger);
    profileKernel(kernelName: string, inputs: NamedTensorMap, f: () => Tensor[]): KernelProfile;
    logKernelProfile(kernelProfile: KernelProfile): void;
}
declare function checkComputationForErrors<D extends DataType>(vals: DataTypeMap[D], dtype: D, kernelName: string): boolean;
declare class Logger {
    logKernelProfile(name: string, result: Tensor, vals: TypedArray, timeMs: number | {
        error: string;
    }, inputs: NamedTensorMap, extraInfo?: string): void;
}

/// <amd-module name="@tensorflow/tfjs-core/dist/profiler_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/io/progress" />
/**
 * Monitor Promise.all progress, fire onProgress callback function.
 *
 * @param promises Promise list going to be monitored
 * @param onProgress Callback function. Fired when a promise resolved.
 * @param startFraction Optional fraction start. Default to 0.
 * @param endFraction Optional fraction end. Default to 1.
 */
declare function monitorPromisesProgress(promises: Array<Promise<{} | void>>, onProgress: OnProgressCallback, startFraction?: number, endFraction?: number): Promise<{}[]>;

/// <amd-module name="@tensorflow/tfjs-core/dist/io/progress_test" />
{ };
/// <amd-module name="@tensorflow/tfjs-core/dist/ops/linalg/qr" />
/**
 * Compute QR decomposition of m-by-n matrix using Householder transformation.
 *
 * Implementation based on
 *   [http://www.cs.cornell.edu/~bindel/class/cs6210-f09/lec18.pdf]
 * (http://www.cs.cornell.edu/~bindel/class/cs6210-f09/lec18.pdf)
 *
 * ```js
 * const a = tf.tensor2d([[1, 2], [3, 4]]);
 * let [q, r] = tf.linalg.qr(a);
 * console.log('Q');
 * q.print();
 * console.log('R');
 * r.print();
 * console.log('Orthogonalized');
 * q.dot(q.transpose()).print()  // should be nearly the identity matrix.
 * console.log('Reconstructed');
 * q.dot(r).print(); // should be nearly [[1, 2], [3, 4]];
 * ```
 *
 * @param x The `tf.Tensor` to be QR-decomposed. Must have rank >= 2. Suppose
 *   it has the shape `[..., M, N]`.
 * @param fullMatrices An optional boolean parameter. Defaults to `false`.
 *   If `true`, compute full-sized `Q`. If `false` (the default),
 *   compute only the leading N columns of `Q` and `R`.
 * @returns An `Array` of two `tf.Tensor`s: `[Q, R]`. `Q` is a unitary matrix,
 *   i.e., its columns all have unit norm and are mutually orthogonal.
 *   If `M >= N`,
 *     If `fullMatrices` is `false` (default),
 *       - `Q` has a shape of `[..., M, N]`,
 *       - `R` has a shape of `[..., N, N]`.
 *     If `fullMatrices` is `true` (default),
 *       - `Q` has a shape of `[..., M, M]`,
 *       - `R` has a shape of `[..., M, N]`.
 *   If `M < N`,
 *     - `Q` has a shape of `[..., M, M]`,
 *     - `R` has a shape of `[..., M, N]`.
 * @throws If the rank of `x` is less than 2.
 *
 * @doc {heading:'Operations',
 *       subheading:'Linear Algebra',
 *       namespace:'linalg'}
 */
declare function qr_(x: Tensor, fullMatrices?: boolean): [Tensor, Tensor];
declare const qr: typeof qr_;
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/linalg/qr_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/ragged_gather" />
/**
 * Gather ragged slices from params axis 0 according to indices.
 *
 * @param paramsNestedSplits: A list of at least 1 Tensor with type 'int32' The
 *     nestedRowSplits tensors that define the row-partitioning for the params
 *     RaggedTensor input.
 * @param paramsDenseValues: A Tensor. The flatValues for the params
 *     RaggedTensor.
 * @param indices: A Tensor. Must be one of type: int32. Indices in the
 *     outermost dimension of params of the values that should be gathered.
 * @param outputRaggedRank: An int that is >= 0. The ragged rank of the output
 *     RaggedTensor. outputNestedSplits will contain this number of rowSplits
 *     tensors. This value should equal indices.shape.ndims + params.raggedRank
 *     - 1.
 * @return A map with the following properties:
 *     - outputNestedSplits: A list of outputRaggedRank Tensor objects with the
 * same type as paramsNestedSplits.
 *     - outputDenseValues: A Tensor. Has the same type as paramsDenseValues.
 * @doc {heading: 'Operations', subheading: 'Ragged'}
 */
interface RaggedGatherMap {
    outputNestedSplits: Tensor[];
    outputDenseValues: Tensor;
}
declare function raggedGather_(paramsNestedSplits: Tensor[], paramsDenseValues: Tensor | TensorLike, indices: Tensor | TensorLike, outputRaggedRank: number): RaggedGatherMap;
declare const raggedGather: typeof raggedGather_;
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/ragged_gather_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/ragged_range" />
/**
 * Returns a RaggedTensor result composed from rtDenseValues and rtNestedSplits,
 * such that result[i] = [starts[i], starts[i] + deltas[i], ..., limits[i]]).
 *
 * @param starts: A Tensor. Must be one of the following types:
 *     'float32', 'int32'. The starts of each range.
 * @param limits: A Tensor. Must have the same type as starts. The limits of
 *     each range.
 * @param deltas: A Tensor. Must have the same type as starts. The deltas of
 *     each range.
 * @return A map with the following properties:
 *     - rtNestedSplits: A Tensor of type 'int32'.
 *     - rtDenseValues: A Tensor. Has the same type as starts.
 */
declare function raggedRange_(starts: Tensor | TensorLike, limits: Tensor | TensorLike, deltas: Tensor | TensorLike): NamedTensorMap;
declare const raggedRange: typeof raggedRange_;
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/ragged_range_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/ragged_tensor_to_tensor" />
/**
 * Create a dense tensor from a ragged tensor, possibly altering its shape.
 *
 * The raggedTensorToTensor op creates a dense tensor from am array of row
 * partition tensors, a value vector, and default values. If the shape is
 * unspecified, the minimal shape required to contain all the elements in the
 * ragged tensor (the natural shape) will be used. If some dimensions are left
 * unspecified, then the size of the natural shape is used in that dimension.
 *
 * The defaultValue will be broadcast to the output shape. After that, the
 * values from the ragged tensor overwrite the default values. Note that the
 * defaultValue must have less dimensions than the value.
 *
 * The row partition tensors are in the order of the dimensions. At present, the
 * types can be: "ROW_SPLITS": the row_splits tensor from the ragged tensor.
 *   "VALUE_ROWIDS": the value_rowids tensor from the ragged tensor.
 *   "FIRST_DIM_SIZE": if value_rowids is used for the first dimension, then it
 * is preceded by "FIRST_DIM_SIZE".
 * ```
 * @param shape: A Tensor. Must be one of the following types: 'int32'. The
 *     desired shape of the output tensor. If left unspecified (empty), the
 *     minimal shape required to contain all the elements in the ragged tensor
 *     (the natural shape) will be used. If some dimensions are left
 *     unspecified, then the size of the natural shape is used in that
 *     dimension.
 *
 *     Note that dense dimensions cannot be modified by the shape argument.
 *     Trying to change the size of a dense dimension will cause the op to fail.
 *     Examples: natural shape: [4, 5, 6] shape: -1 output shape: [4, 5, 6]
 *
 *     natural shape: [4, 5, 6] shape: [3, -1, 2] output shape: [3, 5, 2]
 *
 *     natural shape: [4, 5, 6] shape: [3, 7, 2] output shape: [3, 7, 2]
 * @param values: A Tensor. A 1D tensor representing the values of the ragged
 *     tensor.
 * @param defaultValue: A Tensor. Must have the same type as values. The
 *     defaultValue when the shape is larger than the ragged tensor. The
 *     defaultValue is broadcast until it is the shape of the output tensor,
 *     and then overwritten by values in the ragged tensor. The default value
 *     must be compatible with this broadcast operation, and must have fewer
 *     dimensions than the value tensor.
 * @param rowPartitionTensors: A list of at least 1 Tensor objects with the same
 *     type in: 'int32'.
 * @param rowPartitionTypes: A list of strings. The types of the row partition
 *     tensors. At present, these can be:
 *     "ROW_SPLITS": the row_splits tensor from the ragged tensor.
 *     "VALUE_ROWIDS": the value_rowids tensor from the ragged tensor.
 *     "FIRST_DIM_SIZE": if value_rowids is used for the first dimension, then
 *         it is preceeded by "FIRST_DIM_SIZE". The tensors are in the order of
 *         the dimensions.
 * @return A Tensor. Has the same type as values.
 * @doc {heading: 'Operations', subheading: 'Ragged'}
 */
declare function raggedTensorToTensor_(shape: Tensor | TensorLike, values: Tensor | TensorLike, defaultValue: Tensor | TensorLike, rowPartitionTensors: Tensor[], rowPartitionTypes: string[]): Tensor;
declare const raggedTensorToTensor: typeof raggedTensorToTensor_;
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/ragged_tensor_to_tensor_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/ragged_to_dense_util" />
declare enum RowPartitionType {
    FIRST_DIM_SIZE = 0,
    VALUE_ROWIDS = 1,
    ROW_LENGTHS = 2,
    ROW_SPLITS = 3,
    ROW_LIMITS = 4,
    ROW_STARTS = 5
}
declare function combineRaggedTensorToTensorShapes(raggedRank: number, shape: number[], valueShape: number[]): number[];
declare function getRowPartitionTypesHelper(rowPartitionTypeStrings: string[]): RowPartitionType[];
declare function getRaggedRank(rowPartitionTypes: RowPartitionType[]): number;
declare function validateDefaultValueShape(defaultValueShape: number[], valueShape: number[]): void;

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/rand" />
/**
 * Creates a `tf.Tensor` with values sampled from a random number generator
 * function defined by the user.
 *
 * @param shape An array of integers defining the output tensor shape.
 * @param randFunction A random number generator function which is called
 * for each element in the output tensor.
 * @param dtype The data type of the output tensor. Defaults to 'float32'.
 *
 * @doc {heading: 'Tensors', subheading: 'Random'}
 */
declare function rand_<R extends Rank>(shape: ShapeMap[R], randFunction: () => number, dtype?: DataType): Tensor<R>;
declare const rand: typeof rand_;
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/random_gamma" />
/**
 * Creates a `tf.Tensor` with values sampled from a gamma distribution.
 *
 * ```js
 * tf.randomGamma([2, 2], 1).print();
 * ```
 *
 * @param shape An array of integers defining the output tensor shape.
 * @param alpha The shape parameter of the gamma distribution.
 * @param beta The inverse scale parameter of the gamma distribution. Defaults
 *     to 1.
 * @param dtype The data type of the output. Defaults to float32.
 * @param seed The seed for the random number generator.
 *
 * @doc {heading: 'Tensors', subheading: 'Random'}
 */
declare function randomGamma_<R extends Rank>(shape: ShapeMap[R], alpha: number, beta?: number, dtype?: 'float32' | 'int32', seed?: number): Tensor<R>;
declare const randomGamma: typeof randomGamma_;
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/random_gamma_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/random_normal" />
/**
 * Creates a `tf.Tensor` with values sampled from a normal distribution.
 *
 * ```js
 * tf.randomNormal([2, 2]).print();
 * ```
 *
 * @param shape An array of integers defining the output tensor shape.
 * @param mean The mean of the normal distribution.
 * @param stdDev The standard deviation of the normal distribution.
 * @param dtype The data type of the output.
 * @param seed The seed for the random number generator.
 *
 * @doc {heading: 'Tensors', subheading: 'Random'}
 */
declare function randomNormal_<R extends Rank>(shape: ShapeMap[R], mean?: number, stdDev?: number, dtype?: 'float32' | 'int32', seed?: number): Tensor<R>;
declare const randomNormal: typeof randomNormal_;
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/random_normal_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/random_standard_normal" />
/**
 * Creates a `tf.Tensor` with values sampled from a normal distribution.
 *
 * The generated values will have mean 0 and standard deviation 1.
 *
 * ```js
 * tf.randomStandardNormal([2, 2]).print();
 * ```
 *
 * @param shape An array of integers defining the output tensor shape.
 * @param dtype The data type of the output.
 * @param seed The seed for the random number generator.
 *
 * @doc {heading: 'Tensors', subheading: 'Random'}
 */
declare function randomStandardNormal_<R extends Rank>(shape: ShapeMap[R], dtype?: 'float32' | 'int32', seed?: number): Tensor<R>;
declare const randomStandardNormal: typeof randomStandardNormal_;
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/random_standard_normal_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/random_uniform" />
/**
 * Creates a `tf.Tensor` with values sampled from a uniform distribution.
 *
 * The generated values follow a uniform distribution in the range [minval,
 * maxval). The lower bound minval is included in the range, while the upper
 * bound maxval is excluded.
 *
 * ```js
 * tf.randomUniform([2, 2]).print();
 * ```
 *
 * @param shape An array of integers defining the output tensor shape.
 * @param minval The lower bound on the range of random values to generate.
 *   Defaults to 0.
 * @param maxval The upper bound on the range of random values to generate.
 *   Defaults to 1.
 * @param dtype The data type of the output tensor. Defaults to 'float32'.
 *
 * @doc {heading: 'Tensors', subheading: 'Random'}
 */
declare function randomUniform_<R extends Rank>(shape: ShapeMap[R], minval?: number, maxval?: number, dtype?: DataType, seed?: number | string): Tensor<R>;
declare const randomUniform: typeof randomUniform_;
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/random_uniform_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/rand_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/rand_util" />
interface RandomBase {
    nextValue(): number;
}
interface RandomGamma {
    nextValue(): number;
}
interface RandNormalDataTypes {
    float32: Float32Array;
    int32: Int32Array;
}
interface RandGammaDataTypes {
    float32: Float32Array;
    int32: Int32Array;
}
declare class MPRandGauss implements RandomBase {
    private mean;
    private stdDev;
    private nextVal;
    private dtype?;
    private truncated?;
    private upper?;
    private lower?;
    private random;
    constructor(mean: number, stdDeviation: number, dtype?: keyof RandNormalDataTypes, truncated?: boolean, seed?: number);
    /** Returns next sample from a Gaussian distribution. */
    nextValue(): number;
    /** Handles proper rounding for non-floating-point numbers. */
    private convertValue;
    /** Returns true if less than 2-standard-deviations from the mean. */
    private isValidTruncated;
}
declare class RandGamma implements RandomGamma {
    private alpha;
    private beta;
    private d;
    private c;
    private dtype?;
    private randu;
    private randn;
    constructor(alpha: number, beta: number, dtype: keyof RandGammaDataTypes, seed?: number);
    /** Returns next sample from a gamma distribution. */
    nextValue(): number;
    /** Handles proper rounding for non-floating-point numbers. */
    private convertValue;
}
declare class UniformRandom implements RandomBase {
    private min;
    private range;
    private random;
    private dtype?;
    constructor(min?: number, max?: number, dtype?: keyof RandNormalDataTypes, seed?: string | number);
    /** Handles proper rounding for non floating point numbers. */
    private canReturnFloat;
    private convertValue;
    nextValue(): number;
}
declare function jarqueBeraNormalityTest(values: TypedArray | number[]): void;
declare function expectArrayInMeanStdRange(actual: TypedArray | number[], expectedMean: number, expectedStdDev: number, epsilon?: number): void;

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/range" />
/**
 * Creates a new `tf.Tensor1D` filled with the numbers in the range provided.
 *
 * The tensor is a half-open interval meaning it includes start, but
 * excludes stop. Decrementing ranges and negative step values are also
 * supported.
 *
 *
 * ```js
 * tf.range(0, 9, 2).print();
 * ```
 *
 * @param start An integer start value
 * @param stop An integer stop value
 * @param step An integer increment (will default to 1 or -1)
 * @param dtype The data type of the output tensor. Defaults to 'float32'.
 *
 * @doc {heading: 'Tensors', subheading: 'Creation'}
 */
declare function range(start: number, stop: number, step?: number, dtype?: 'float32' | 'int32'): Tensor1D;

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/range_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-data/dist/readers" />
/**
 * Create a `CSVDataset` by reading and decoding CSV file(s) from provided URL
 * or local path if it's in Node environment.
 *
 * Note: If isLabel in columnConfigs is `true` for at least one column, the
 * element in returned `CSVDataset` will be an object of
 * `{xs:features, ys:labels}`: xs is a dict of features key/value pairs, ys
 * is a dict of labels key/value pairs. If no column is marked as label,
 * returns a dict of features only.
 *
 * ```js
 * const csvUrl =
 * 'https://storage.googleapis.com/tfjs-examples/multivariate-linear-regression/data/boston-housing-train.csv';
 *
 * async function run() {
 *   // We want to predict the column "medv", which represents a median value of
 *   // a home (in $1000s), so we mark it as a label.
 *   const csvDataset = tf.data.csv(
 *     csvUrl, {
 *       columnConfigs: {
 *         medv: {
 *           isLabel: true
 *         }
 *       }
 *     });
 *
 *   // Number of features is the number of column names minus one for the label
 *   // column.
 *   const numOfFeatures = (await csvDataset.columnNames()).length - 1;
 *
 *   // Prepare the Dataset for training.
 *   const flattenedDataset =
 *     csvDataset
 *     .map(({xs, ys}) =>
 *       {
 *         // Convert xs(features) and ys(labels) from object form (keyed by
 *         // column name) to array form.
 *         return {xs:Object.values(xs), ys:Object.values(ys)};
 *       })
 *     .batch(10);
 *
 *   // Define the model.
 *   const model = tf.sequential();
 *   model.add(tf.layers.dense({
 *     inputShape: [numOfFeatures],
 *     units: 1
 *   }));
 *   model.compile({
 *     optimizer: tf.train.sgd(0.000001),
 *     loss: 'meanSquaredError'
 *   });
 *
 *   // Fit the model using the prepared Dataset
 *   return model.fitDataset(flattenedDataset, {
 *     epochs: 10,
 *     callbacks: {
 *       onEpochEnd: async (epoch, logs) => {
 *         console.log(epoch + ':' + logs.loss);
 *       }
 *     }
 *   });
 * }
 *
 * await run();
 * ```
 *
 * @param source URL or local path to get CSV file. If it's a local path, it
 * must have prefix `file://` and it only works in node environment.
 * @param csvConfig (Optional) A CSVConfig object that contains configurations
 *     of reading and decoding from CSV file(s).
 *
 * @doc {
 *   heading: 'Data',
 *   subheading: 'Creation',
 *   namespace: 'data',
 *   configParamIndices: [1]
 *  }
 */
declare function csv(source: RequestInfo, csvConfig?: CSVConfig): CSVDataset;
/**
 * Create a `Dataset` that produces each element by calling a provided function.
 *
 * Note that repeated iterations over this `Dataset` may produce different
 * results, because the function will be called anew for each element of each
 * iteration.
 *
 * Also, beware that the sequence of calls to this function may be out of order
 * in time with respect to the logical order of the Dataset. This is due to the
 * asynchronous lazy nature of stream processing, and depends on downstream
 * transformations (e.g. .shuffle()). If the provided function is pure, this is
 * no problem, but if it is a closure over a mutable state (e.g., a traversal
 * pointer), then the order of the produced elements may be scrambled.
 *
 * ```js
 * let i = -1;
 * const func = () =>
 *    ++i < 5 ? {value: i, done: false} : {value: null, done: true};
 * const ds = tf.data.func(func);
 * await ds.forEachAsync(e => console.log(e));
 * ```
 *
 * @param f A function that produces one data element on each call.
 */
declare function func<T extends TensorContainer>(f: () => IteratorResult<T> | Promise<IteratorResult<T>>): Dataset<T>;
/**
 * Create a `Dataset` that produces each element from provided JavaScript
 * generator, which is a function*
 * (https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide/Iterators_and_Generators#Generator_functions),
 * or a function that returns an
 * iterator
 * (https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide/Iterators_and_Generators#Generator_functions).
 *
 * The returned iterator should have `.next()` function that returns element in
 * format of `{value: TensorContainer, done:boolean}`.
 *
 * Example of creating a dataset from an iterator factory:
 * ```js
 * function makeIterator() {
 *   const numElements = 10;
 *   let index = 0;
 *
 *   const iterator = {
 *     next: () => {
 *       let result;
 *       if (index < numElements) {
 *         result = {value: index, done: false};
 *         index++;
 *         return result;
 *       }
 *       return {value: index, done: true};
 *     }
 *   };
 *   return iterator;
 * }
 * const ds = tf.data.generator(makeIterator);
 * await ds.forEachAsync(e => console.log(e));
 * ```
 *
 * Example of creating a dataset from a generator:
 * ```js
 * function* dataGenerator() {
 *   const numElements = 10;
 *   let index = 0;
 *   while (index < numElements) {
 *     const x = index;
 *     index++;
 *     yield x;
 *   }
 * }
 *
 * const ds = tf.data.generator(dataGenerator);
 * await ds.forEachAsync(e => console.log(e));
 * ```
 *
 * @param generator A JavaScript generator function that returns a JavaScript
 *     iterator.
 *
 * @doc {
 *   heading: 'Data',
 *   subheading: 'Creation',
 *   namespace: 'data',
 *   configParamIndices: [1]
 *  }
 */
declare function generator<T extends TensorContainer>(generator: () => Iterator<T> | Promise<Iterator<T>>): Dataset<T>;
/**
 * Create an iterator that generates `Tensor`s from webcam video stream. This
 * API only works in Browser environment when the device has webcam.
 *
 * Note: this code snippet only works when the device has a webcam. It will
 * request permission to open the webcam when running.
 * ```js
 * const videoElement = document.createElement('video');
 * videoElement.width = 100;
 * videoElement.height = 100;
 * const cam = await tf.data.webcam(videoElement);
 * const img = await cam.capture();
 * img.print();
 * cam.stop();
 * ```
 *
 * @param webcamVideoElement A `HTMLVideoElement` used to play video from
 *     webcam. If this element is not provided, a hidden `HTMLVideoElement` will
 *     be created. In that case, `resizeWidth` and `resizeHeight` must be
 *     provided to set the generated tensor shape.
 * @param webcamConfig A `WebcamConfig` object that contains configurations of
 *     reading and manipulating data from webcam video stream.
 *
 * @doc {
 *   heading: 'Data',
 *   subheading: 'Creation',
 *   namespace: 'data',
 *   ignoreCI: true
 *  }
 */
declare function webcam(webcamVideoElement?: HTMLVideoElement, webcamConfig?: WebcamConfig): Promise<WebcamIterator>;
/**
 * Create an iterator that generates frequency-domain spectrogram `Tensor`s from
 * microphone audio stream with browser's native FFT. This API only works in
 * browser environment when the device has microphone.
 *
 * Note: this code snippet only works when the device has a microphone. It will
 * request permission to open the microphone when running.
 * ```js
 * const mic = await tf.data.microphone({
 *   fftSize: 1024,
 *   columnTruncateLength: 232,
 *   numFramesPerSpectrogram: 43,
 *   sampleRateHz:44100,
 *   includeSpectrogram: true,
 *   includeWaveform: true
 * });
 * const audioData = await mic.capture();
 * const spectrogramTensor = audioData.spectrogram;
 * spectrogramTensor.print();
 * const waveformTensor = audioData.waveform;
 * waveformTensor.print();
 * mic.stop();
 * ```
 *
 * @param microphoneConfig A `MicrophoneConfig` object that contains
 *     configurations of reading audio data from microphone.
 *
 * @doc {
 *   heading: 'Data',
 *   subheading: 'Creation',
 *   namespace: 'data',
 *   ignoreCI: true
 *  }
 */
declare function microphone(microphoneConfig?: MicrophoneConfig): Promise<MicrophoneIterator>;

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/real" />
/**
 * Returns the real part of a complex (or real) tensor.
 *
 * Given a tensor input, this operation returns a tensor of type float that is
 * the real part of each element in input considered as a complex number.
 *
 * If the input is real, it simply makes a clone.
 *
 * ```js
 * const x = tf.complex([-2.25, 3.25], [4.75, 5.75]);
 * tf.real(x).print();
 * ```
 *
 * @doc {heading: 'Tensors', subheading: 'Creation'}
 */
declare function real_<T extends Tensor>(input: T | TensorLike): T;
declare const real: typeof real_;
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/gradients/RealDiv_grad" />
declare const divGradConfig: GradConfig;

/// <amd-module name="@tensorflow/tfjs-core/dist/public/chained_ops/reciprocal" />
declare module '../../tensor' {
    interface Tensor<R extends Rank = Rank> {
        reciprocal<T extends Tensor>(this: T): T;
    }
}

/// <amd-module name="@tensorflow/tfjs-core/dist/gradients/Reciprocal_grad" />
declare const reciprocalGradConfig: GradConfig;

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/reciprocal_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-layers/dist/layers/recurrent" />
/**
 * TensorFlow.js Layers: Recurrent Neural Network Layers.
 */
/**
 * Standardize `apply()` args to a single list of tensor inputs.
 *
 * When running a model loaded from file, the input tensors `initialState` and
 * `constants` are passed to `RNN.apply()` as part of `inputs` instead of the
 * dedicated kwargs fields. `inputs` consists of
 * `[inputs, initialState0, initialState1, ..., constant0, constant1]` in this
 * case.
 * This method makes sure that arguments are
 * separated and that `initialState` and `constants` are `Array`s of tensors
 * (or None).
 *
 * @param inputs Tensor or `Array` of  tensors.
 * @param initialState Tensor or `Array` of tensors or `null`/`undefined`.
 * @param constants Tensor or `Array` of tensors or `null`/`undefined`.
 * @returns An object consisting of
 *   inputs: A tensor.
 *   initialState: `Array` of tensors or `null`.
 *   constants: `Array` of tensors or `null`.
 * @throws ValueError, if `inputs` is an `Array` but either `initialState` or
 *   `constants` is provided.
 */
declare function standardizeArgs(inputs: Tensor | Tensor[] | SymbolicTensor | SymbolicTensor[], initialState: Tensor | Tensor[] | SymbolicTensor | SymbolicTensor[], constants: Tensor | Tensor[] | SymbolicTensor | SymbolicTensor[], numConstants?: number): {
    inputs: Tensor | SymbolicTensor;
    initialState: Tensor[] | SymbolicTensor[];
    constants: Tensor[] | SymbolicTensor[];
};
/**
 * Iterates over the time dimension of a tensor.
 *
 * @param stepFunction RNN step function.
 *   Parameters:
 *     inputs: tensor with shape `[samples, ...]` (no time dimension),
 *       representing input for the batch of samples at a certain time step.
 *     states: an Array of tensors.
 *   Returns:
 *     outputs: tensor with shape `[samples, outputDim]` (no time dimension).
 *     newStates: list of tensors, same length and shapes as `states`. The first
 *       state in the list must be the output tensor at the previous timestep.
 * @param inputs Tensor of temporal data of shape `[samples, time, ...]` (at
 *   least 3D).
 * @param initialStates Tensor with shape `[samples, outputDim]` (no time
 *   dimension), containing the initial values of the states used in the step
 *   function.
 * @param goBackwards If `true`, do the iteration over the time dimension in
 *   reverse order and return the reversed sequence.
 * @param mask Binary tensor with shape `[sample, time, 1]`, with a zero for
 *   every element that is masked.
 * @param constants An Array of constant values passed at each step.
 * @param unroll Whether to unroll the RNN or to use a symbolic loop. *Not*
 *   applicable to this imperative deeplearn.js backend. Its value is ignored.
 * @param needPerStepOutputs Whether the per-step outputs are to be
 *   concatenated into a single tensor and returned (as the second return
 *   value). Default: `false`. This arg is included so that the relatively
 *   expensive concatenation of the stepwise outputs can be omitted unless
 *   the stepwise outputs need to be kept (e.g., for an LSTM layer of which
 *   `returnSequence` is `true`.)
 * @returns An Array: `[lastOutput, outputs, newStates]`.
 *   lastOutput: the lastest output of the RNN, of shape `[samples, ...]`.
 *   outputs: tensor with shape `[samples, time, ...]` where each entry
 *     `output[s, t]` is the output of the step function at time `t` for sample
 *     `s`. This return value is provided if and only if the
 *     `needPerStepOutputs` is set as `true`. If it is set as `false`, this
 *     return value will be `undefined`.
 *   newStates: Array of tensors, latest states returned by the step function,
 *      of shape `(samples, ...)`.
 * @throws ValueError If input dimension is less than 3.
 *
 * TODO(nielsene): This needs to be tidy-ed.
 */
declare function rnn(stepFunction: RnnStepFunction, inputs: Tensor, initialStates: Tensor[], goBackwards?: boolean, mask?: Tensor, constants?: Tensor[], unroll?: boolean, needPerStepOutputs?: boolean): [Tensor, Tensor, Tensor[]];
declare interface BaseRNNLayerArgs extends LayerArgs {
    /**
     * A RNN cell instance. A RNN cell is a class that has:
     *   - a `call()` method, which takes `[Tensor, Tensor]` as the
     *     first input argument. The first item is the input at time t, and
     *     second item is the cell state at time t.
     *     The `call()` method returns `[outputAtT, statesAtTPlus1]`.
     *     The `call()` method of the cell can also take the argument `constants`,
     *     see section "Note on passing external constants" below.
     *     Porting Node: PyKeras overrides the `call()` signature of RNN cells,
     *       which are Layer subtypes, to accept two arguments. tfjs-layers does
     *       not do such overriding. Instead we preseve the `call()` signature,
     *       which due to its `Tensor|Tensor[]` argument and return value is
     *       flexible enough to handle the inputs and states.
     *   - a `stateSize` attribute. This can be a single integer (single state)
     *     in which case it is the size of the recurrent state (which should be
     *     the same as the size of the cell output). This can also be an Array of
     *     integers (one size per state). In this case, the first entry
     *     (`stateSize[0]`) should be the same as the size of the cell output.
     * It is also possible for `cell` to be a list of RNN cell instances, in which
     * case the cells get stacked on after the other in the RNN, implementing an
     * efficient stacked RNN.
     */
    cell?: RNNCell | RNNCell[];
    /**
     * Whether to return the last output in the output sequence, or the full
     * sequence.
     */
    returnSequences?: boolean;
    /**
     * Whether to return the last state in addition to the output.
     */
    returnState?: boolean;
    /**
     * If `true`, process the input sequence backwards and return the reversed
     * sequence (default: `false`).
     */
    goBackwards?: boolean;
    /**
     * If `true`, the last state for each sample at index i in a batch will be
     * used as initial state of the sample of index i in the following batch
     * (default: `false`).
     *
     * You can set RNN layers to be "stateful", which means that the states
     * computed for the samples in one batch will be reused as initial states
     * for the samples in the next batch. This assumes a one-to-one mapping
     * between samples in different successive batches.
     *
     * To enable "statefulness":
     *   - specify `stateful: true` in the layer constructor.
     *   - specify a fixed batch size for your model, by passing
     *     - if sequential model:
     *       `batchInputShape: [...]` to the first layer in your model.
     *     - else for functional model with 1 or more Input layers:
     *       `batchShape: [...]` to all the first layers in your model.
     *     This is the expected shape of your inputs
     *     *including the batch size*.
     *     It should be a tuple of integers, e.g., `[32, 10, 100]`.
     *   - specify `shuffle: false` when calling `LayersModel.fit()`.
     *
     * To reset the state of your model, call `resetStates()` on either the
     * specific layer or on the entire model.
     */
    stateful?: boolean;
    /**
     * If `true`, the network will be unrolled, else a symbolic loop will be
     * used. Unrolling can speed up a RNN, although it tends to be more
     * memory-intensive. Unrolling is only suitable for short sequences (default:
     * `false`).
     * Porting Note: tfjs-layers has an imperative backend. RNNs are executed with
     *   normal TypeScript control flow. Hence this property is inapplicable and
     *   ignored in tfjs-layers.
     */
    unroll?: boolean;
    /**
     * Dimensionality of the input (integer).
     *   This option (or alternatively, the option `inputShape`) is required when
     *   this layer is used as the first layer in a model.
     */
    inputDim?: number;
    /**
     * Length of the input sequences, to be specified when it is constant.
     * This argument is required if you are going to connect `Flatten` then
     * `Dense` layers upstream (without it, the shape of the dense outputs cannot
     * be computed). Note that if the recurrent layer is not the first layer in
     * your model, you would need to specify the input length at the level of the
     * first layer (e.g., via the `inputShape` option).
     */
    inputLength?: number;
}
declare class RNN extends Layer {
    /** @nocollapse */
    static className: string;
    readonly cell: RNNCell;
    readonly returnSequences: boolean;
    readonly returnState: boolean;
    readonly goBackwards: boolean;
    readonly unroll: boolean;
    stateSpec: InputSpec[];
    protected states_: Tensor[];
    protected keptStates: Tensor[][];
    private numConstants;
    constructor(args: RNNLayerArgs);
    getStates(): Tensor[];
    setStates(states: Tensor[]): void;
    computeOutputShape(inputShape: Shape | Shape[]): Shape | Shape[];
    computeMask(inputs: Tensor | Tensor[], mask?: Tensor | Tensor[]): Tensor | Tensor[];
    /**
     * Get the current state tensors of the RNN.
     *
     * If the state hasn't been set, return an array of `null`s of the correct
     * length.
     */
    get states(): Tensor[];
    set states(s: Tensor[]);
    build(inputShape: Shape | Shape[]): void;
    /**
     * Reset the state tensors of the RNN.
     *
     * If the `states` argument is `undefined` or `null`, will set the
     * state tensor(s) of the RNN to all-zero tensors of the appropriate
     * shape(s).
     *
     * If `states` is provided, will set the state tensors of the RNN to its
     * value.
     *
     * @param states Optional externally-provided initial states.
     * @param training Whether this call is done during training. For stateful
     *   RNNs, this affects whether the old states are kept or discarded. In
     *   particular, if `training` is `true`, the old states will be kept so
     *   that subsequent backpropgataion through time (BPTT) may work properly.
     *   Else, the old states will be discarded.
     */
    resetStates(states?: Tensor | Tensor[], training?: boolean): void;
    apply(inputs: Tensor | Tensor[] | SymbolicTensor | SymbolicTensor[], kwargs?: Kwargs): Tensor | Tensor[] | SymbolicTensor | SymbolicTensor[];
    call(inputs: Tensor | Tensor[], kwargs: Kwargs): Tensor | Tensor[];
    getInitialState(inputs: Tensor): Tensor[];
    get trainableWeights(): LayerVariable[];
    get nonTrainableWeights(): LayerVariable[];
    setFastWeightInitDuringBuild(value: boolean): void;
    getConfig(): serialization.ConfigDict;
    /** @nocollapse */
    static fromConfig<T extends serialization.Serializable>(cls: serialization.SerializableConstructor<T>, config: serialization.ConfigDict, customObjects?: serialization.ConfigDict): T;
}
/**
 * An RNNCell layer.
 *
 * @doc {heading: 'Layers', subheading: 'Classes'}
 */
declare abstract class RNNCell extends Layer {
    /**
     * Size(s) of the states.
     * For RNN cells with only a single state, this is a single integer.
     */
    abstract stateSize: number | number[];
    dropoutMask: Tensor | Tensor[];
    recurrentDropoutMask: Tensor | Tensor[];
}
declare interface SimpleRNNCellLayerArgs extends LayerArgs {
    /**
     * units: Positive integer, dimensionality of the output space.
     */
    units: number;
    /**
     * Activation function to use.
     * Default: hyperbolic tangent ('tanh').
     * If you pass `null`,  'linear' activation will be applied.
     */
    activation?: ActivationIdentifier;
    /**
     * Whether the layer uses a bias vector.
     */
    useBias?: boolean;
    /**
     * Initializer for the `kernel` weights matrix, used for the linear
     * transformation of the inputs.
     */
    kernelInitializer?: InitializerIdentifier | Initializer;
    /**
     * Initializer for the `recurrentKernel` weights matrix, used for
     * linear transformation of the recurrent state.
     */
    recurrentInitializer?: InitializerIdentifier | Initializer;
    /**
     * Initializer for the bias vector.
     */
    biasInitializer?: InitializerIdentifier | Initializer;
    /**
     * Regularizer function applied to the `kernel` weights matrix.
     */
    kernelRegularizer?: RegularizerIdentifier | Regularizer;
    /**
     * Regularizer function applied to the `recurrent_kernel` weights matrix.
     */
    recurrentRegularizer?: RegularizerIdentifier | Regularizer;
    /**
     * Regularizer function applied to the bias vector.
     */
    biasRegularizer?: RegularizerIdentifier | Regularizer;
    /**
     * Constraint function applied to the `kernel` weights matrix.
     */
    kernelConstraint?: ConstraintIdentifier | Constraint;
    /**
     * Constraint function applied to the `recurrentKernel` weights matrix.
     */
    recurrentConstraint?: ConstraintIdentifier | Constraint;
    /**
     * Constraint function applied to the bias vector.
     */
    biasConstraint?: ConstraintIdentifier | Constraint;
    /**
     * Float number between 0 and 1. Fraction of the units to drop for the linear
     * transformation of the inputs.
     */
    dropout?: number;
    /**
     * Float number between 0 and 1. Fraction of the units to drop for the linear
     * transformation of the recurrent state.
     */
    recurrentDropout?: number;
    /**
     * This is added for test DI purpose.
     */
    dropoutFunc?: Function;
}
declare class SimpleRNNCell extends RNNCell {
    /** @nocollapse */
    static className: string;
    readonly units: number;
    readonly activation: Activation;
    readonly useBias: boolean;
    readonly kernelInitializer: Initializer;
    readonly recurrentInitializer: Initializer;
    readonly biasInitializer: Initializer;
    readonly kernelConstraint: Constraint;
    readonly recurrentConstraint: Constraint;
    readonly biasConstraint: Constraint;
    readonly kernelRegularizer: Regularizer;
    readonly recurrentRegularizer: Regularizer;
    readonly biasRegularizer: Regularizer;
    readonly dropout: number;
    readonly recurrentDropout: number;
    readonly dropoutFunc: Function;
    readonly stateSize: number;
    kernel: LayerVariable;
    recurrentKernel: LayerVariable;
    bias: LayerVariable;
    readonly DEFAULT_ACTIVATION = "tanh";
    readonly DEFAULT_KERNEL_INITIALIZER = "glorotNormal";
    readonly DEFAULT_RECURRENT_INITIALIZER = "orthogonal";
    readonly DEFAULT_BIAS_INITIALIZER: InitializerIdentifier;
    constructor(args: SimpleRNNCellLayerArgs);
    build(inputShape: Shape | Shape[]): void;
    call(inputs: Tensor | Tensor[], kwargs: Kwargs): Tensor | Tensor[];
    getConfig(): serialization.ConfigDict;
}
declare interface SimpleRNNLayerArgs extends BaseRNNLayerArgs {
    /**
     * Positive integer, dimensionality of the output space.
     */
    units: number;
    /**
     * Activation function to use.
     *
     * Defaults to  hyperbolic tangent (`tanh`)
     *
     * If you pass `null`, no activation will be applied.
     */
    activation?: ActivationIdentifier;
    /**
     * Whether the layer uses a bias vector.
     */
    useBias?: boolean;
    /**
     * Initializer for the `kernel` weights matrix, used for the linear
     * transformation of the inputs.
     */
    kernelInitializer?: InitializerIdentifier | Initializer;
    /**
     * Initializer for the `recurrentKernel` weights matrix, used for
     * linear transformation of the recurrent state.
     */
    recurrentInitializer?: InitializerIdentifier | Initializer;
    /**
     * Initializer for the bias vector.
     */
    biasInitializer?: InitializerIdentifier | Initializer;
    /**
     * Regularizer function applied to the kernel weights matrix.
     */
    kernelRegularizer?: RegularizerIdentifier | Regularizer;
    /**
     * Regularizer function applied to the recurrentKernel weights matrix.
     */
    recurrentRegularizer?: RegularizerIdentifier | Regularizer;
    /**
     * Regularizer function applied to the bias vector.
     */
    biasRegularizer?: RegularizerIdentifier | Regularizer;
    /**
     * Constraint function applied to the kernel weights matrix.
     */
    kernelConstraint?: ConstraintIdentifier | Constraint;
    /**
     * Constraint function applied to the recurrentKernel weights matrix.
     */
    recurrentConstraint?: ConstraintIdentifier | Constraint;
    /**
     * Constraint function applied to the bias vector.
     */
    biasConstraint?: ConstraintIdentifier | Constraint;
    /**
     * Number between 0 and 1. Fraction of the units to drop for the linear
     * transformation of the inputs.
     */
    dropout?: number;
    /**
     * Number between 0 and 1. Fraction of the units to drop for the linear
     * transformation of the recurrent state.
     */
    recurrentDropout?: number;
    /**
     * This is added for test DI purpose.
     */
    dropoutFunc?: Function;
}
/**
 * RNNLayerConfig is identical to BaseRNNLayerConfig, except it makes the
 * `cell` property required. This interface is to be used with constructors
 * of concrete RNN layer subtypes.
 */
declare interface RNNLayerArgs extends BaseRNNLayerArgs {
    cell: RNNCell | RNNCell[];
}
declare class SimpleRNN extends RNN {
    /** @nocollapse */
    static className: string;
    constructor(args: SimpleRNNLayerArgs);
    call(inputs: Tensor | Tensor[], kwargs: Kwargs): Tensor | Tensor[];
    /** @nocollapse */
    static fromConfig<T extends serialization.Serializable>(cls: serialization.SerializableConstructor<T>, config: serialization.ConfigDict): T;
}
declare interface GRUCellLayerArgs extends SimpleRNNCellLayerArgs {
    /**
     * Activation function to use for the recurrent step.
     *
     * Defaults to hard sigmoid (`hardSigmoid`).
     *
     * If `null`, no activation is applied.
     */
    recurrentActivation?: ActivationIdentifier;
    /**
     * Implementation mode, either 1 or 2.
     *
     * Mode 1 will structure its operations as a larger number of
     *   smaller dot products and additions.
     *
     * Mode 2 will batch them into fewer, larger operations. These modes will
     * have different performance profiles on different hardware and
     * for different applications.
     *
     * Note: For superior performance, TensorFlow.js always uses implementation
     * 2, regardless of the actual value of this configuration field.
     */
    implementation?: number;
    /**
     * GRU convention (whether to apply reset gate after or before matrix
     * multiplication). false = "before", true = "after" (only false is
     * supported).
     */
    resetAfter?: boolean;
}
declare class GRUCell extends RNNCell {
    /** @nocollapse */
    static className: string;
    readonly units: number;
    readonly activation: Activation;
    readonly recurrentActivation: Activation;
    readonly useBias: boolean;
    readonly kernelInitializer: Initializer;
    readonly recurrentInitializer: Initializer;
    readonly biasInitializer: Initializer;
    readonly kernelRegularizer: Regularizer;
    readonly recurrentRegularizer: Regularizer;
    readonly biasRegularizer: Regularizer;
    readonly kernelConstraint: Constraint;
    readonly recurrentConstraint: Constraint;
    readonly biasConstraint: Constraint;
    readonly dropout: number;
    readonly recurrentDropout: number;
    readonly dropoutFunc: Function;
    readonly stateSize: number;
    readonly implementation: number;
    readonly DEFAULT_ACTIVATION = "tanh";
    readonly DEFAULT_RECURRENT_ACTIVATION: ActivationIdentifier;
    readonly DEFAULT_KERNEL_INITIALIZER = "glorotNormal";
    readonly DEFAULT_RECURRENT_INITIALIZER = "orthogonal";
    readonly DEFAULT_BIAS_INITIALIZER: InitializerIdentifier;
    kernel: LayerVariable;
    recurrentKernel: LayerVariable;
    bias: LayerVariable;
    constructor(args: GRUCellLayerArgs);
    build(inputShape: Shape | Shape[]): void;
    call(inputs: Tensor | Tensor[], kwargs: Kwargs): Tensor | Tensor[];
    getConfig(): serialization.ConfigDict;
}
declare interface GRULayerArgs extends SimpleRNNLayerArgs {
    /**
     * Activation function to use for the recurrent step.
     *
     * Defaults to hard sigmoid (`hardSigmoid`).
     *
     * If `null`, no activation is applied.
     */
    recurrentActivation?: ActivationIdentifier;
    /**
     * Implementation mode, either 1 or 2.
     *
     * Mode 1 will structure its operations as a larger number of
     * smaller dot products and additions.
     *
     * Mode 2 will batch them into fewer, larger operations. These modes will
     * have different performance profiles on different hardware and
     * for different applications.
     *
     * Note: For superior performance, TensorFlow.js always uses implementation
     * 2, regardless of the actual value of this configuration field.
     */
    implementation?: number;
}
declare class GRU extends RNN {
    /** @nocollapse */
    static className: string;
    constructor(args: GRULayerArgs);
    call(inputs: Tensor | Tensor[], kwargs: Kwargs): Tensor | Tensor[];
    /** @nocollapse */
    static fromConfig<T extends serialization.Serializable>(cls: serialization.SerializableConstructor<T>, config: serialization.ConfigDict): T;
}
declare interface LSTMCellLayerArgs extends SimpleRNNCellLayerArgs {
    /**
     * Activation function to use for the recurrent step.
     *
     * Defaults to hard sigmoid (`hardSigmoid`).
     *
     * If `null`, no activation is applied.
     */
    recurrentActivation?: ActivationIdentifier;
    /**
     * If `true`, add 1 to the bias of the forget gate at initialization.
     * Setting it to `true` will also force `biasInitializer = 'zeros'`.
     * This is recommended in
     * [Jozefowicz et
     * al.](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
     */
    unitForgetBias?: boolean;
    /**
     * Implementation mode, either 1 or 2.
     *
     * Mode 1 will structure its operations as a larger number of
     *   smaller dot products and additions.
     *
     * Mode 2 will batch them into fewer, larger operations. These modes will
     * have different performance profiles on different hardware and
     * for different applications.
     *
     * Note: For superior performance, TensorFlow.js always uses implementation
     * 2, regardless of the actual value of this configuration field.
     */
    implementation?: number;
}
declare class LSTMCell extends RNNCell {
    /** @nocollapse */
    static className: string;
    readonly units: number;
    readonly activation: Activation;
    readonly recurrentActivation: Activation;
    readonly useBias: boolean;
    readonly kernelInitializer: Initializer;
    readonly recurrentInitializer: Initializer;
    readonly biasInitializer: Initializer;
    readonly unitForgetBias: boolean;
    readonly kernelConstraint: Constraint;
    readonly recurrentConstraint: Constraint;
    readonly biasConstraint: Constraint;
    readonly kernelRegularizer: Regularizer;
    readonly recurrentRegularizer: Regularizer;
    readonly biasRegularizer: Regularizer;
    readonly dropout: number;
    readonly recurrentDropout: number;
    readonly dropoutFunc: Function;
    readonly stateSize: number[];
    readonly implementation: number;
    readonly DEFAULT_ACTIVATION = "tanh";
    readonly DEFAULT_RECURRENT_ACTIVATION = "hardSigmoid";
    readonly DEFAULT_KERNEL_INITIALIZER = "glorotNormal";
    readonly DEFAULT_RECURRENT_INITIALIZER = "orthogonal";
    readonly DEFAULT_BIAS_INITIALIZER = "zeros";
    kernel: LayerVariable;
    recurrentKernel: LayerVariable;
    bias: LayerVariable;
    constructor(args: LSTMCellLayerArgs);
    build(inputShape: Shape | Shape[]): void;
    call(inputs: Tensor | Tensor[], kwargs: Kwargs): Tensor | Tensor[];
    getConfig(): serialization.ConfigDict;
}
declare interface LSTMLayerArgs extends SimpleRNNLayerArgs {
    /**
     * Activation function to use for the recurrent step.
     *
     * Defaults to hard sigmoid (`hardSigmoid`).
     *
     * If `null`, no activation is applied.
     */
    recurrentActivation?: ActivationIdentifier;
    /**
     * If `true`, add 1 to the bias of the forget gate at initialization.
     * Setting it to `true` will also force `biasInitializer = 'zeros'`.
     * This is recommended in
     * [Jozefowicz et
     * al.](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
     */
    unitForgetBias?: boolean;
    /**
     * Implementation mode, either 1 or 2.
     *   Mode 1 will structure its operations as a larger number of
     *   smaller dot products and additions, whereas mode 2 will
     *   batch them into fewer, larger operations. These modes will
     *   have different performance profiles on different hardware and
     *   for different applications.
     *
     * Note: For superior performance, TensorFlow.js always uses implementation
     * 2, regardless of the actual value of this config field.
     */
    implementation?: number;
}
declare class LSTM extends RNN {
    /** @nocollapse */
    static className: string;
    constructor(args: LSTMLayerArgs);
    call(inputs: Tensor | Tensor[], kwargs: Kwargs): Tensor | Tensor[];
    /** @nocollapse */
    static fromConfig<T extends serialization.Serializable>(cls: serialization.SerializableConstructor<T>, config: serialization.ConfigDict): T;
}
declare interface StackedRNNCellsArgs extends LayerArgs {
    /**
     * An `Array` of `RNNCell` instances.
     */
    cells: RNNCell[];
}
declare class StackedRNNCells extends RNNCell {
    /** @nocollapse */
    static className: string;
    protected cells: RNNCell[];
    constructor(args: StackedRNNCellsArgs);
    get stateSize(): number[];
    call(inputs: Tensor | Tensor[], kwargs: Kwargs): Tensor | Tensor[];
    build(inputShape: Shape | Shape[]): void;
    getConfig(): serialization.ConfigDict;
    /** @nocollapse */
    static fromConfig<T extends serialization.Serializable>(cls: serialization.SerializableConstructor<T>, config: serialization.ConfigDict, customObjects?: serialization.ConfigDict): T;
    get trainableWeights(): LayerVariable[];
    get nonTrainableWeights(): LayerVariable[];
    /**
     * Retrieve the weights of a the model.
     *
     * @returns A flat `Array` of `tf.Tensor`s.
     */
    getWeights(): Tensor[];
    /**
     * Set the weights of the model.
     *
     * @param weights An `Array` of `tf.Tensor`s with shapes and types matching
     *     the output of `getWeights()`.
     */
    setWeights(weights: Tensor[]): void;
}
declare function generateDropoutMask(args: {
    ones: () => tfc.Tensor;
    rate: number;
    training?: boolean;
    count?: number;
    dropoutFunc?: Function;
}): tfc.Tensor | tfc.Tensor[];

/// <amd-module name="@tensorflow/tfjs-layers/dist/keras_format/layers/recurrent_serialization" />
interface BaseRNNLayerConfig extends LayerConfig {
    cell?: RNNCellSerialization | RNNCellSerialization[];
    return_sequences?: boolean;
    return_state?: boolean;
    go_backwards?: boolean;
    stateful?: boolean;
    unroll?: boolean;
    input_dim?: number;
    input_length?: number;
}
interface SimpleRNNCellConfig extends LayerConfig {
    units: number;
    activation?: ActivationSerialization;
    use_bias?: boolean;
    kernel_initializer?: InitializerSerialization;
    recurrent_initializer?: InitializerSerialization;
    bias_initializer?: InitializerSerialization;
    kernel_regularizer?: RegularizerSerialization;
    recurrent_regularizer?: RegularizerSerialization;
    bias_regularizer?: RegularizerSerialization;
    kernel_constraint?: ConstraintSerialization;
    recurrent_constraint?: ConstraintSerialization;
    bias_constraint?: ConstraintSerialization;
    dropout?: number;
    recurrent_dropout?: number;
}
declare type SimpleRNNCellSerialization = BaseSerialization<'SimpleRNNCell', SimpleRNNCellConfig>;
interface SimpleRNNLayerConfig extends BaseRNNLayerConfig {
    units: number;
    activation?: ActivationSerialization;
    use_bias?: boolean;
    kernel_initializer?: InitializerSerialization;
    recurrent_initializer?: InitializerSerialization;
    bias_initializer?: InitializerSerialization;
    kernel_regularizer?: RegularizerSerialization;
    recurrent_regularizer?: RegularizerSerialization;
    bias_regularizer?: RegularizerSerialization;
    kernel_constraint?: ConstraintSerialization;
    recurrent_constraint?: ConstraintSerialization;
    bias_constraint?: ConstraintSerialization;
    dropout?: number;
    recurrent_dropout?: number;
}
declare type SimpleRNNLayerSerialization = BaseLayerSerialization<'SimpleRNN', SimpleRNNLayerConfig>;
interface GRUCellConfig extends SimpleRNNCellConfig {
    recurrent_activation?: string;
    implementation?: number;
}
declare type GRUCellSerialization = BaseSerialization<'GRUCell', GRUCellConfig>;
interface GRULayerConfig extends SimpleRNNLayerConfig {
    recurrent_activation?: ActivationSerialization;
    implementation?: number;
}
declare type GRULayerSerialization = BaseLayerSerialization<'GRU', GRULayerConfig>;
interface LSTMCellConfig extends SimpleRNNCellConfig {
    recurrent_activation?: ActivationSerialization;
    unit_forget_bias?: boolean;
    implementation?: number;
}
declare type LSTMCellSerialization = BaseSerialization<'LSTMCell', LSTMCellConfig>;
interface LSTMLayerConfig extends SimpleRNNLayerConfig {
    recurrent_activation?: ActivationSerialization;
    unit_forget_bias?: boolean;
    implementation?: number;
}
declare type LSTMLayerSerialization = BaseLayerSerialization<'LSTM', LSTMLayerConfig>;
interface StackedRNNCellsConfig extends LayerConfig {
    cells: RNNCellSerialization[];
}
declare type StackedRNNCellsSerialization = BaseSerialization<'StackedRNNCells', StackedRNNCellsConfig>;
declare type RNNCellSerialization = SimpleRNNCellSerialization | GRUCellSerialization | LSTMCellSerialization | StackedRNNCellsSerialization;
declare type RecurrentLayerSerialization = SimpleRNNLayerSerialization | LSTMLayerSerialization | GRULayerSerialization;
declare type RecurrentLayerClassName = RecurrentLayerSerialization['class_name'];
/**
 * A string array of valid RecurrentLayer class names.
 *
 * This is guaranteed to match the `RecurrentLayerClassName` union type.
 */
declare const recurrentLayerClassNames: RecurrentLayerClassName[];

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/reduce_util" />
declare const PARALLELIZE_THRESHOLD = 30;
interface ReduceInfo {
    windowSize: number;
    batchSize: number;
    inSize: number;
    outSize: number;
}
declare function computeOptimalWindowSize(inSize: number): number;

/// <amd-module name="@tensorflow/tfjs-core/dist/public/chained_ops/register_all_chained_ops" />

/// <amd-module name="@tensorflow/tfjs-core/dist/public/chained_ops/register_all_chained_ops_test" />
/// <amd-module name="@tensorflow/tfjs-core/dist/register_all_gradients" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/optimizers/register_optimizers" />
declare function registerOptimizers(): void;

/// <amd-module name="@tensorflow/tfjs-layers/dist/regularizers" />
/**
 * Regularizer base class.
 */
declare abstract class Regularizer extends serialization.Serializable {
    abstract apply(x: Tensor): Scalar;
}
interface L1L2Args {
    /** L1 regularization rate. Defaults to 0.01. */
    l1?: number;
    /** L2 regularization rate. Defaults to 0.01. */
    l2?: number;
}
interface L1Args {
    /** L1 regularization rate. Defaults to 0.01. */
    l1: number;
}
interface L2Args {
    /** L2 regularization rate. Defaults to 0.01. */
    l2: number;
}
declare class L1L2 extends Regularizer {
    /** @nocollapse */
    static className: string;
    private readonly l1;
    private readonly l2;
    private readonly hasL1;
    private readonly hasL2;
    constructor(args?: L1L2Args);
    /**
     * Porting note: Renamed from __call__.
     * @param x Variable of which to calculate the regularization score.
     */
    apply(x: Tensor): Scalar;
    getConfig(): serialization.ConfigDict;
    /** @nocollapse */
    static fromConfig<T extends serialization.Serializable>(cls: serialization.SerializableConstructor<T>, config: serialization.ConfigDict): T;
}
declare function l1(args?: L1Args): L1L2;
declare function l2(args: L2Args): L1L2;
/** @docinline */
declare type RegularizerIdentifier = 'l1l2' | string;
declare const REGULARIZER_IDENTIFIER_REGISTRY_SYMBOL_MAP: {
    [identifier in RegularizerIdentifier]: string;
};
declare function serializeRegularizer(constraint: Regularizer): serialization.ConfigDictValue;
declare function deserializeRegularizer(config: serialization.ConfigDict, customObjects?: serialization.ConfigDict): Regularizer;
declare function getRegularizer(identifier: RegularizerIdentifier | serialization.ConfigDict | Regularizer): Regularizer;

/// <amd-module name="@tensorflow/tfjs-layers/dist/keras_format/regularizer_config" />
declare type L1L2Config = {
    l1?: number;
    l2?: number;
};
declare type L1L2Serialization = BaseSerialization<'L1L2', L1L2Config>;
declare type RegularizerSerialization = L1L2Serialization;
declare type RegularizerClassName = RegularizerSerialization['class_name'];
/**
 * A string array of valid Regularizer class names.
 *
 * This is guaranteed to match the `RegularizerClassName` union type.
 */
declare const regularizerClassNames: RegularizerClassName[];
/// <amd-module name="@tensorflow/tfjs-core/dist/public/chained_ops/relu" />
declare module '../../tensor' {
    interface Tensor<R extends Rank = Rank> {
        relu<T extends Tensor>(): T;
    }
}

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/relu6" />
/**
 * Computes rectified linear 6 element-wise: `min(max(x, 0), 6)`.
 *
 * ```js
 * const x = tf.tensor1d([-1, 2, -3, 8]);
 *
 * x.relu6().print();  // or tf.relu6(x)
 * ```
 * @param x The input tensor. If the dtype is `bool`, the output dtype will be
 *     `int32`.
 *
 * @doc {heading: 'Operations', subheading: 'Basic math'}
 */
declare function relu6_<T extends Tensor>(x: T | TensorLike): T;
declare const relu6: typeof relu6_;
{ };
/// <amd-module name="@tensorflow/tfjs-core/dist/gradients/Relu6_grad" />
declare const relu6GradConfig: GradConfig;

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/relu6_test" />
{ };
/// <amd-module name="@tensorflow/tfjs-core/dist/gradients/Relu_grad" />
declare const reluGradConfig: GradConfig;

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/relu_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/reshape" />
/**
 * Reshapes a `tf.Tensor` to a given shape.
 *
 * Given an input tensor, returns a new tensor with the same values as the
 * input tensor with shape `shape`.
 *
 * If one component of shape is the special value -1, the size of that
 * dimension is computed so that the total size remains constant. In
 * particular, a shape of [-1] flattens into 1-D. At most one component of
 * shape can be -1.
 *
 * If shape is 1-D or higher, then the operation returns a tensor with shape
 * shape filled with the values of tensor. In this case, the number of
 * elements implied by shape must be the same as the number of elements in
 * tensor.
 *
 * ```js
 * const x = tf.tensor1d([1, 2, 3, 4]);
 * x.reshape([2, 2]).print();
 * ```
 *
 * @param x The input tensor to be reshaped.
 * @param shape An array of integers defining the output tensor shape.
 *
 * @doc {heading: 'Tensors', subheading: 'Transformations'}
 */
declare function reshape_<R extends Rank>(x: Tensor | TensorLike, shape: ShapeMap[R]): Tensor<R>;
declare const reshape: typeof reshape_;
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/public/chained_ops/reshape_as" />
declare module '../../tensor' {
    interface Tensor<R extends Rank = Rank> {
        reshapeAs<T extends Tensor>(x: T): T;
    }
}
/// <amd-module name="@tensorflow/tfjs-core/dist/gradients/Reshape_grad" />
declare const reshapeGradConfig: GradConfig;
/// <amd-module name="@tensorflow/tfjs-core/dist/gradients/ResizeBilinear_grad" />
declare const resizeBilinearGradConfig: GradConfig;
/// <amd-module name="@tensorflow/tfjs-core/dist/gradients/ResizeNearestNeighbor_grad" />
declare const resizeNearestNeighborGradConfig: GradConfig;

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/image/resize_bilinear" />
/**
 * Bilinear resize a single 3D image or a batch of 3D images to a new shape.
 *
 * @param images The images, of rank 4 or rank 3, of shape
 *     `[batch, height, width, inChannels]`. If rank 3, batch of 1 is assumed.
 * @param size The new shape `[newHeight, newWidth]` to resize the
 *     images to. Each channel is resized individually.
 * @param alignCorners Defaults to `false`. If true, rescale
 *     input by `(new_height - 1) / (height - 1)`, which exactly aligns the 4
 *     corners of images and resized images. If false, rescale by
 *     `new_height / height`. Treat similarly the width dimension.
 * @param halfPixelCenters Defaults to `false`. Whether to assume pixel centers
 *     are at 0.5, which would make the floating point coordinates of the top
 *     left pixel 0.5, 0.5.
 *
 * @doc {heading: 'Operations', subheading: 'Images', namespace: 'image'}
 */
declare function resizeBilinear_<T extends Tensor3D | Tensor4D>(images: T | TensorLike, size: [number, number], alignCorners?: boolean, halfPixelCenters?: boolean): T;
declare const resizeBilinear: typeof resizeBilinear_;
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/image/resize_bilinear_test" />
{ };
/// <amd-module name="@tensorflow/tfjs-core/dist/public/chained_ops/resize_nearest_neighbor" />
declare module '../../tensor' {
    interface Tensor<R extends Rank = Rank> {
        resizeNearestNeighbor<T extends Tensor3D | Tensor4D>(newShape2D: [number, number], alignCorners?: boolean, halfFloatCenters?: boolean): T;
    }
}

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/image/resize_nearest_neighbor_test" />
{ };
/// <amd-module name="@tensorflow/tfjs-core/dist/public/chained_ops/reverse" />
declare module '../../tensor' {
    interface Tensor<R extends Rank = Rank> {
        reverse<T extends Tensor>(this: T, axis?: number | number[]): T;
    }
}

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/reverse_1d" />
/**
 * Reverses a `tf.Tensor1D`.
 *
 * @param x The input tensor.
 */
declare function reverse1d_(x: Tensor1D | TensorLike): Tensor1D;
declare const reverse1d: typeof reverse1d_;
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/reverse_1d_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/reverse_2d" />
/**
 * Reverses a `tf.Tensor2D` along a specified axis.
 *
 * @param x The input tensor.
 * @param axis The set of dimensions to reverse. Must be in the
 *     range [-rank(x), rank(x)). Defaults to all axes.
 */
declare function reverse2d_(x: Tensor2D | TensorLike, axis?: number | number[]): Tensor2D;
declare const reverse2d: typeof reverse2d_;
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/reverse_2d_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/reverse_3d" />
/**
 * Reverses a `tf.Tensor3D` along a specified axis.
 *
 * @param x The input tensor.
 * @param axis The set of dimensions to reverse. Must be in the
 *     range [-rank(x), rank(x)). Defaults to all axes.
 */
declare function reverse3d_(x: Tensor3D | TensorLike, axis?: number | number[]): Tensor3D;
declare const reverse3d: typeof reverse3d_;
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/reverse_3d_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/reverse_4d" />
/**
 * Reverses a `tf.Tensor4D` along a specified axis.
 *
 * @param x The input tensor.
 * @param axis The set of dimensions to reverse. Must be in the
 *     range [-rank(x), rank(x)). Defaults to all axes.
 */
declare function reverse4d_(x: Tensor4D | TensorLike, axis?: number | number[]): Tensor4D;
declare const reverse4d: typeof reverse4d_;
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/reverse_4d_test" />
{ };
/// <amd-module name="@tensorflow/tfjs-core/dist/gradients/Reverse_grad" />
declare const reverseGradConfig: GradConfig;

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/reverse_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/spectral/rfft" />
/**
 * Real value input fast Fourier transform.
 *
 * Computes the 1-dimensional discrete Fourier transform over the
 * inner-most dimension of the real input.
 *
 * ```js
 * const real = tf.tensor1d([1, 2, 3]);
 *
 * real.rfft().print();
 * ```
 * @param input The real value input to compute an rfft over.
 *
 * @doc {heading: 'Operations', subheading: 'Spectral', namespace: 'spectral'}
 */
declare function rfft_(input: Tensor, fftLength?: number): Tensor;
declare const rfft: typeof rfft_;
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/spectral/rfft_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-data/dist/util/ring_buffer" />
/**
 * A ring buffer, providing O(1) FIFO, LIFO, and related operations.
 */
declare class RingBuffer<T> {
    capacity: number;
    protected begin: number;
    protected end: number;
    protected doubledCapacity: number;
    protected data: T[];
    /**
     * Constructs a `RingBuffer`.
     * @param capacity The number of items that the buffer can accomodate.
     */
    constructor(capacity: number);
    /**
     * Map any index into the range 0 <= index < 2*capacity.
     */
    protected wrap(index: number): number;
    protected get(index: number): T;
    protected set(index: number, value: T): void;
    /**
     * Returns the current number of items in the buffer.
     */
    length(): number;
    /**
     * Reports whether the buffer is full.
     * @returns true if the number of items in the buffer equals its capacity, and
     *   false otherwise.
     */
    isFull(): boolean;
    /**
     * Reports whether the buffer is empty.
     * @returns true if the number of items in the buffer equals zero, and
     *   false otherwise.
     */
    isEmpty(): boolean;
    /**
     * Adds an item to the end of the buffer.
     */
    push(value: T): void;
    /**
     * Adds many items to the end of the buffer, in order.
     */
    pushAll(values: T[]): void;
    /**
     * Removes and returns the last item in the buffer.
     */
    pop(): T;
    /**
     * Adds an item to the beginning of the buffer.
     */
    unshift(value: T): void;
    /**
     * Removes and returns the first item in the buffer.
     */
    shift(): T;
    /**
     * Removes and returns a specific item in the buffer, and moves the last item
     * to the vacated slot.  This is useful for implementing a shuffling stream.
     * Note that this operation necessarily scrambles the original order.
     *
     * @param relativeIndex: the index of the item to remove, relative to the
     *   first item in the buffer (e.g., hiding the ring nature of the underlying
     *   storage).
     */
    shuffleExcise(relativeIndex: number): T;
}

/// <amd-module name="@tensorflow/tfjs-core/dist/optimizers/rmsprop_optimizer" />
/** @doclink Optimizer */
declare class RMSPropOptimizer extends Optimizer {
    protected learningRate: number;
    protected decay: number;
    protected momentum: number;
    protected epsilon: number;
    /** @nocollapse */
    static get className(): string;
    private centered;
    private accumulatedMeanSquares;
    private accumulatedMoments;
    private accumulatedMeanGrads;
    constructor(learningRate: number, decay?: number, momentum?: number, epsilon?: number, centered?: boolean);
    applyGradients(variableGradients: NamedTensorMap | NamedTensor[]): void;
    dispose(): void;
    getWeights(): Promise<NamedTensor[]>;
    setWeights(weightValues: NamedTensor[]): Promise<void>;
    getConfig(): ConfigDict;
    /** @nocollapse */
    static fromConfig<T extends Serializable>(cls: SerializableConstructor<T>, config: ConfigDict): T;
}

/// <amd-module name="@tensorflow/tfjs-core/dist/optimizers/rmsprop_optimizer_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/rotate_util" />
declare function getImageCenter(center: number | [number, number], imageHeight: number, imageWidth: number): [number, number];

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/image/rotate_with_offset" />
/**
 * Rotates the input image tensor counter-clockwise with an optional offset
 * center of rotation. Currently available in the CPU, WebGL, and WASM backends.
 *
 * @param image 4d tensor of shape `[batch, imageHeight, imageWidth, depth]`.
 * @param radians The amount of rotation.
 * @param fillValue The value to fill in the empty space leftover
 *     after rotation. Can be either a single grayscale value (0-255), or an
 *     array of three numbers `[red, green, blue]` specifying the red, green,
 *     and blue channels. Defaults to `0` (black).
 * @param center The center of rotation. Can be either a single value (0-1), or
 *     an array of two numbers `[centerX, centerY]`. Defaults to `0.5` (rotates
 *     the image around its center).
 *
 * @doc {heading: 'Operations', subheading: 'Images', namespace: 'image'}
 */
declare function rotateWithOffset_(image: Tensor4D | TensorLike, radians: number, fillValue?: number | [number, number, number], center?: number | [number, number]): Tensor4D;
declare const rotateWithOffset: typeof rotateWithOffset_;
{ };
/// <amd-module name="@tensorflow/tfjs-core/dist/ops/image/rotate_with_offset_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/round" />
/**
 * Computes round of input `tf.Tensor` element-wise: `round(x)`.
 * It implements banker's rounding.
 *
 * ```js
 * const x = tf.tensor1d([.6, 1.1, -3.3]);
 *
 * x.round().print();  // or tf.round(x)
 * ```
 * @param x The input tensor.
 *
 * @doc {heading: 'Operations', subheading: 'Basic math'}
 */
declare function round_<T extends Tensor>(x: T | TensorLike): T;
declare const round: typeof round_;
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/gradients/Round_grad" />
declare const roundGradConfig: GradConfig;

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/round_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/io/router_registry" />
declare type IORouter = (url: string | string[], loadOptions?: LoadOptions) => IOHandler;
declare class IORouterRegistry {
    private static instance;
    private saveRouters;
    private loadRouters;
    private constructor();
    private static getInstance;
    /**
     * Register a save-handler router.
     *
     * @param saveRouter A function that maps a URL-like string onto an instance
     * of `IOHandler` with the `save` method defined or `null`.
     */
    static registerSaveRouter(saveRouter: IORouter): void;
    /**
     * Register a load-handler router.
     *
     * @param loadRouter A function that maps a URL-like string onto an instance
     * of `IOHandler` with the `load` method defined or `null`.
     */
    static registerLoadRouter(loadRouter: IORouter): void;
    /**
     * Look up IOHandler for saving, given a URL-like string.
     *
     * @param url
     * @returns If only one match is found, an instance of IOHandler with the
     * `save` method defined. If no match is found, `null`.
     * @throws Error, if more than one match is found.
     */
    static getSaveHandlers(url: string | string[]): IOHandler[];
    /**
     * Look up IOHandler for loading, given a URL-like string.
     *
     * @param url
     * @param loadOptions Optional, custom load options.
     * @returns All valid handlers for `url`, given the currently registered
     *   handler routers.
     */
    static getLoadHandlers(url: string | string[], loadOptions?: LoadOptions): IOHandler[];
    private static getHandlers;
}
declare const registerSaveRouter: (loudRouter: IORouter) => void;
declare const registerLoadRouter: (loudRouter: IORouter) => void;
declare const getSaveHandlers: (url: string | string[]) => IOHandler[];
declare const getLoadHandlers: (url: string | string[], loadOptions?: LoadOptions) => IOHandler[];

/// <amd-module name="@tensorflow/tfjs-core/dist/io/router_registry_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/public/chained_ops/rsqrt" />
declare module '../../tensor' {
    interface Tensor<R extends Rank = Rank> {
        rsqrt<T extends Tensor>(this: T): T;
    }
}

/// <amd-module name="@tensorflow/tfjs-core/dist/gradients/Rsqrt_grad" />
declare const rsqrtGradConfig: GradConfig;

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/rsqrt_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/scalar" />
/**
 * Creates rank-0 `tf.Tensor` (scalar) with the provided value and dtype.
 *
 * The same functionality can be achieved with `tf.tensor`, but in general
 * we recommend using `tf.scalar` as it makes the code more readable.
 *
 * ```js
 * tf.scalar(3.14).print();
 * ```
 *
 * @param value The value of the scalar.
 * @param dtype The data type.
 *
 * @doc {heading: 'Tensors', subheading: 'Creation'}
 */
declare function scalar(value: number | boolean | string | Uint8Array, dtype?: DataType): Scalar;

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/scatter_nd" />
/**
 * Creates a new tensor by applying sparse updates to individual
 * values or slices within a zero tensor of the given shape tensor according to
 * indices. This operator is the inverse of the `tf.gatherND` operator which
 * extracts values or slices from a given tensor.
 *
 * ```js
 * const indices = tf.tensor2d([4, 3, 1, 7], [4, 1], 'int32');
 * const updates = tf.tensor1d([9, 10, 11, 12]);
 * const shape = [8];
 * tf.scatterND(indices, updates, shape).print() //[0, 11, 0, 10, 9, 0, 0, 12]
 * ```
 *
 * @param indices The tensor contains the indices into the output tensor.
 * @param updates The tensor contains the value for the indices.
 * @param shape: The shape of the output tensor.
 *
 * @doc {heading: 'Operations', subheading: 'Slicing and Joining'}
 */
declare function scatterND_<R extends Rank>(indices: Tensor | TensorLike, updates: Tensor | TensorLike, shape: ShapeMap[R]): Tensor<R>;
declare const scatterND: typeof scatterND_;
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/scatter_nd_test" />
{ };
/// <amd-module name="@tensorflow/tfjs-core/dist/ops/scatter_nd_util" />

/**
 * Check whether updates.shape = indices.shape[:batchDim] +
 * shape[sliceDim:]
 *
 * @param x The input tensor.
 */
declare function validateUpdateShape(shape: number[], indices: Tensor, updates: Tensor): void;
interface ScatterShapeInfo {
    sliceRank: number;
    numUpdates: number;
    sliceSize: number;
    strides: number[];
    outputSize: number;
}
/**
 * Validate scatter nd inputs.
 *
 * @param update The tensor contains the update values.
 * @param indices The tensor contains the indices for the update values.
 * @param shape The shape of the output tensor.
 */
declare function validateInput(updates: Tensor, indices: Tensor, shape: number[]): void;
/**
 * Calculate the shape information for the output.
 *
 * @param update The tensor contains the update values.
 * @param indices The tensor contains the indices for the update values.
 * @param shape The shape of the output tensor.
 *
 * @returns ScatterShapeInfo
 */
declare function calculateShapes(updates: TensorInfo, indices: TensorInfo, shape: number[]): ScatterShapeInfo;

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/search_sorted" />
/**
 * Searches for where a value would go in a sorted sequence.
 *
 * This is not a method for checking containment (like javascript in).
 *
 * The typical use case for this operation is "binning", "bucketing", or
 * "discretizing". The values are assigned to bucket-indices based on the edges
 * listed in 'sortedSequence'. This operation returns the bucket-index for each
 * value.
 *
 * The side argument controls which index is returned if a value lands exactly
 * on an edge.
 *
 * The axis is not settable for this operation. It always operates on the
 * innermost dimension (axis=-1). The operation will accept any number of outer
 * dimensions.
 *
 * Note: This operation assumes that 'sortedSequence' is sorted along the
 * innermost axis, maybe using 'sort(..., axis=-1)'. If the sequence is not
 * sorted no error is raised and the content of the returned tensor is not well
 * defined.
 *
 * ```js
 * const edges = tf.tensor1d([-1, 3.3, 9.1, 10.0]);
 * let values = tf.tensor1d([0.0, 4.1, 12.0]);
 * const result1 = tf.searchSorted(edges, values, 'left');
 * result1.print(); // [1, 2, 4]
 *
 * const seq = tf.tensor1d([0, 3, 9, 10, 10]);
 * values = tf.tensor1d([0, 4, 10]);
 * const result2 = tf.searchSorted(seq, values, 'left');
 * result2.print(); // [0, 2, 3]
 * const result3 = tf.searchSorted(seq, values, 'right');
 * result3.print(); // [1, 2, 5]
 *
 * const sortedSequence = tf.tensor2d([[0., 3., 8., 9., 10.],
 *                                     [1., 2., 3., 4., 5.]]);
 * values = tf.tensor2d([[9.8, 2.1, 4.3],
 *                       [0.1, 6.6, 4.5, ]]);
 * const result4 = tf.searchSorted(sortedSequence, values, 'left');
 * result4.print(); // [[4, 1, 2], [0, 5, 4]]
 * ```
 * @param sortedSequence: N-D. Sorted sequence.
 * @param values: N-D. Search values.
 * @param side: 'left'|'right'. Defaults to 'left'. 'left' corresponds to lower
 *     bound and 'right' to upper bound.
 * @return An N-D int32 tensor the size of values containing the result of
 *     applying either lower bound or upper bound (depending on side) to each
 *     value. The result is not a global index to the entire Tensor, but the
 *     index in the last dimension.
 * @doc {heading: 'Operations', subheading: 'Evaluation'}
 */
declare function searchSorted_(sortedSequence: Tensor | TensorLike, values: Tensor | TensorLike, side?: 'left' | 'right'): Tensor;
declare const searchSorted: typeof searchSorted_;
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/search_sorted_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/segment_util" />
interface SegOpInfo {
    windowSize: number;
    batchSize: number;
    inSize: number;
    numSegments: number;
}
declare function segOpComputeOptimalWindowSize(inSize: number, numSegments: number): number;
declare function computeOutShape(aShape: number[], axis: number, numSegments: number): number[];
interface GatherOpShapeInfo {
    batchSize: number;
    sliceSize: number;
    outerSize: number;
    dimSize: number;
    outputShape: number[];
}
declare function collectGatherOpShapeInfo(x: TensorInfo, indices: TensorInfo, axis: number, batchDims: number): GatherOpShapeInfo;

/// <amd-module name="@tensorflow/tfjs-core/dist/gradients/Select_grad" />
declare const selectGradConfig: GradConfig;

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/selu" />
/**
 * Computes scaled exponential linear element-wise.
 *
 * `x < 0 ? scale * alpha * (exp(x) - 1) : scale * x`
 *
 * ```js
 * const x = tf.tensor1d([-1, 2, -3, 4]);
 *
 * x.selu().print();  // or tf.selu(x)
 * ```
 * @param x The input tensor.
 *
 * @doc {heading: 'Operations', subheading: 'Basic math'}
 */
declare function selu_<T extends Tensor>(x: T | TensorLike): T;
declare const selu: typeof selu_;
{ };
/// <amd-module name="@tensorflow/tfjs-core/dist/gradients/Selu_grad" />
declare const seluGradConfig: GradConfig;

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/selu_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/selu_util" />
declare const SELU_SCALEALPHA = 1.7580993408473768;
declare const SELU_SCALE = 1.0507009873554805;
/// <amd-module name="@tensorflow/tfjs-core/dist/ops/separable_conv2d" />

/**
 * 2-D convolution with separable filters.
 *
 * Performs a depthwise convolution that acts separately on channels followed
 * by a pointwise convolution that mixes channels. Note that this is
 * separability between dimensions [1, 2] and 3, not spatial separability
 * between dimensions 1 and 2.
 *
 * See
 * [https://www.tensorflow.org/api_docs/python/tf/nn/separable_conv2d](
 *     https://www.tensorflow.org/api_docs/python/tf/nn/separable_conv2d)
 * for more details.
 *
 * @param x The input tensor, of rank 4 or rank 3, of shape
 *     `[batch, height, width, inChannels]`. If rank 3, batch of 1 is
 * assumed.
 * @param depthwiseFilter The depthwise filter tensor, rank 4, of shape
 *     `[filterHeight, filterWidth, inChannels, channelMultiplier]`. This is
 *     the filter used in the first step.
 * @param pointwiseFilter The pointwise filter tensor, rank 4, of shape
 *     `[1, 1, inChannels * channelMultiplier, outChannels]`. This is
 *     the filter used in the second step.
 * @param strides The strides of the convolution: `[strideHeight,
 * strideWidth]`. If strides is a single number, then `strideHeight ==
 * strideWidth`.
 * @param pad The type of padding algorithm.
 *   - `same` and stride 1: output will be of same size as input,
 *       regardless of filter size.
 *   - `valid`: output will be smaller than input if filter is larger
 *       than 1x1.
 *   - For more info, see this guide:
 *     [https://www.tensorflow.org/api_docs/python/tf/nn/convolution](
 *          https://www.tensorflow.org/api_docs/python/tf/nn/convolution)
 * @param dilations The dilation rates: `[dilationHeight, dilationWidth]`
 *     in which we sample input values across the height and width dimensions
 *     in atrous convolution. Defaults to `[1, 1]`. If `rate` is a single
 *     number, then `dilationHeight == dilationWidth`. If it is greater than
 *     1, then all values of `strides` must be 1.
 * @param dataFormat: An optional string from: "NHWC", "NCHW". Defaults to
 *     "NHWC". Specify the data format of the input and output data. With the
 *     default format "NHWC", the data is stored in the order of: [batch,
 *     height, width, channels]. Only "NHWC" is currently supported.
 *
 * @doc {heading: 'Operations', subheading: 'Convolution'}
 */
declare function separableConv2d_<T extends Tensor3D | Tensor4D>(x: T | TensorLike, depthwiseFilter: Tensor4D | TensorLike, pointwiseFilter: Tensor4D | TensorLike, strides: [number, number] | number, pad: 'valid' | 'same', dilation?: [number, number] | number, dataFormat?: 'NHWC' | 'NCHW'): T;
declare const separableConv2d: typeof separableConv2d_;
{ };

/// <amd-module name="@tensorflow/tfjs-layers/dist/layers/serialization" />
/**
 * Instantiate a layer from a config dictionary.
 * @param config dict of the form {class_name: str, config: dict}
 * @param customObjects dict mapping class names (or function names)
 *   of custom (non-Keras) objects to class/functions
 * @param fastWeightInit Optional flag to use fast weight initialization
 *   during deserialization. This is applicable to cases in which
 *   the initialization will be immediately overwritten by loaded weight
 *   values. Default: `false`.
 * @returns Layer instance (may be LayersModel, Sequential, Layer...)
 */
declare function deserialize(config: serialization.ConfigDict, customObjects?: serialization.ConfigDict, fastWeightInit?: boolean): serialization.Serializable;

/// <amd-module name="@tensorflow/tfjs-core/dist/serialization_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-layers/dist/utils/serialization_utils" />
/**
 * Convert a Pythonic config object to TypeScript config object.
 * @param pythonicConfig The config object to convert.
 * @param key Optional key name of the object being converted.
 * @returns Result of the conversion.
 */
declare function convertPythonicToTs(pythonicConfig: PyJsonValue, key?: string): serialization.ConfigDictValue;
/**
 * Convert a TypeScript config object to Python config object.
 * @param tsConfig The config object to convert.
 * @param key Optional key name of the object being converted.
 * @returns Result of the conversion.
 */
declare function convertTsToPythonic(tsConfig: serialization.ConfigDictValue, key?: string): PyJsonValue;
/// <amd-module name="@tensorflow/tfjs-core/dist/ops/setdiff1d_async" />

/**
 * Computes the difference between two lists of numbers.
 *
 * Given a Tensor `x` and a Tensor `y`, this operation returns a Tensor `out`
 * that represents all values that are in `x` but not in `y`. The returned
 * Tensor `out` is sorted in the same order that the numbers appear in `x`
 * (duplicates are preserved). This operation also returns a Tensor indices that
 * represents the position of each out element in `x`. In other words:
 *
 * `out[i] = x[idx[i]] for i in [0, 1, ..., out.length - 1]`
 *
 * ```js
 * const x = [1, 2, 3, 4, 5, 6];
 * const y = [1, 3, 5];
 *
 * const [out, indices] = await tf.setdiff1dAsync(x, y);
 * out.print(); // [2, 4, 6]
 * indices.print(); // [1, 3, 5]
 * ```
 *
 * @param x 1-D Tensor. Values to keep.
 * @param y 1-D Tensor. Must have the same type as x. Values to exclude in the
 *     output.
 * @returns Promise of Tensor tuple [out, indices].
 *  out: Tensor with the same type as x.
 *  indices: A Tensor of type int32.
 *
 * @doc {heading: 'Tensors', subheading: 'Transformations'}
 */
declare function setdiff1dAsync_(x: Tensor | TensorLike, y: Tensor | TensorLike): Promise<[Tensor, Tensor]>;
declare const setdiff1dAsync: typeof setdiff1dAsync_;
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/setdiff1d_async_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/optimizers/sgd_optimizer" />
/** @doclink Optimizer */
declare class SGDOptimizer extends Optimizer {
    protected learningRate: number;
    /** @nocollapse */
    static get className(): string;
    protected c: Scalar;
    constructor(learningRate: number);
    applyGradients(variableGradients: NamedTensorMap | NamedTensor[]): void;
    /**
     * Sets the learning rate of the optimizer.
     */
    setLearningRate(learningRate: number): void;
    dispose(): void;
    getWeights(): Promise<NamedTensor[]>;
    setWeights(weightValues: NamedTensor[]): Promise<void>;
    getConfig(): ConfigDict;
    /** @nocollapse */
    static fromConfig<T extends Serializable>(cls: SerializableConstructor<T>, config: ConfigDict): T;
}

/// <amd-module name="@tensorflow/tfjs-core/dist/optimizers/sgd_optimizer_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/sigmoid" />
/**
 * Computes sigmoid element-wise, `1 / (1 + exp(-x))`
 *
 * ```js
 * const x = tf.tensor1d([0, -1, 2, -3]);
 *
 * x.sigmoid().print();  // or tf.sigmoid(x)
 * ```
 * @param x The input tensor.
 *
 * @doc {heading: 'Operations', subheading: 'Basic math'}
 */
declare function sigmoid_<T extends Tensor>(x: T | TensorLike): T;
declare const sigmoid: typeof sigmoid_;
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/losses/sigmoid_cross_entropy" />
/**
 * Computes the sigmoid cross entropy loss between two tensors.
 *
 * If labelSmoothing is nonzero, smooth the labels towards 1/2:
 *
 *   newMulticlassLabels = multiclassLabels * (1 - labelSmoothing)
 *                         + 0.5 * labelSmoothing
 *
 * @param multiClassLabels The ground truth output tensor of shape
 * [batch_size, num_classes], same dimensions as 'predictions'.
 * @param logits The predicted outputs.
 * @param weights Tensor whose rank is either 0, or the same rank as
 *    `labels`, and must be broadcastable to `labels` (i.e., all dimensions
 *    must be either `1`, or the same as the corresponding `losses`
 *    dimension).
 * @param labelSmoothing If greater than 0, then smooth the labels.
 * @param reduction Type of reduction to apply to loss. Should be of type
 *    `Reduction`
 *
 * @doc { heading: 'Training', subheading: 'Losses', namespace: 'losses' }
 */
declare function sigmoidCrossEntropy_<T extends Tensor, O extends Tensor>(multiClassLabels: T | TensorLike, logits: T | TensorLike, weights?: Tensor | TensorLike, labelSmoothing?: number, reduction?: Reduction): O;
declare const sigmoidCrossEntropy: typeof sigmoidCrossEntropy_;
{ };
/// <amd-module name="@tensorflow/tfjs-core/dist/ops/losses/sigmoid_cross_entropy_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/gradients/Sigmoid_grad" />
declare const sigmoidGradConfig: GradConfig;

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/sigmoid_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/sign" />
/**
 * Returns an element-wise indication of the sign of a number.
 *
 * ```js
 * const x = tf.tensor1d([.6, 1.1, -3.3, NaN, 0]);
 *
 * x.sign().print();  // or tf.sign(x)
 * ```
 * @param x The input Tensor.
 *
 * @doc {heading: 'Operations', subheading: 'Basic math'}
 */
declare function sign_<T extends Tensor>(x: T | TensorLike): T;
declare const sign: typeof sign_;
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/signal_ops_util" />
declare function enclosingPowerOfTwo(value: number): number;
declare function cosineWindow(windowLength: number, a: number, b: number): Tensor1D;

/// <amd-module name="@tensorflow/tfjs-core/dist/gradients/Sign_grad" />
declare const signGradConfig: GradConfig;

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/sign_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/public/chained_ops/sin" />
declare module '../../tensor' {
    interface Tensor<R extends Rank = Rank> {
        sin<T extends Tensor>(this: T): T;
    }
}

/// <amd-module name="@tensorflow/tfjs-core/dist/public/chained_ops/sinh" />
declare module '../../tensor' {
    interface Tensor<R extends Rank = Rank> {
        sinh<T extends Tensor>(this: T): T;
    }
}

/// <amd-module name="@tensorflow/tfjs-core/dist/gradients/Sinh_grad" />
declare const sinhGradConfig: GradConfig;

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/sinh_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/gradients/Sin_grad" />
declare const sinGradConfig: GradConfig;

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/sin_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/public/chained_ops/slice" />
declare module '../../tensor' {
    interface Tensor<R extends Rank = Rank> {
        slice<T extends Tensor>(this: T, begin: number | number[], size?: number | number[]): T;
    }
}

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/slice1d" />
/**
 * Extracts a 1D slice from 1D array starting at coordinates `begin` and is
 * of length `size`. See `slice` for details.
 */
declare function slice1d_(x: Tensor1D | TensorLike, begin: number, size: number): Tensor1D;
declare const slice1d: typeof slice1d_;
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/slice1d_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/slice2d" />
/**
 * Extracts a 2D slice from a 2D array starting at coordinates `begin` and
 * is of size `size`. See `slice` for details.
 */
declare function slice2d_(x: Tensor2D | TensorLike, begin: [number, number], size: [number, number]): Tensor2D;
declare const slice2d: typeof slice2d_;
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/slice2d_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/slice3d" />
/**
 * Extracts a 3D slice from a 3D array starting at coordinates `begin` and
 * is of size `size`. See `slice` for details.
 */
declare function slice3d_(x: Tensor3D | TensorLike, begin: [number, number, number], size: [number, number, number]): Tensor3D;
declare const slice3d: typeof slice3d_;
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/slice3d_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/slice4d" />
/**
 * Extracts a 4D slice from a 4D array starting at coordinates `begin` and
 * is of size `size`. See `slice` for details.
 */
declare function slice4d_(x: Tensor4D | TensorLike, begin: [number, number, number, number], size: [number, number, number, number]): Tensor4D;
declare const slice4d: typeof slice4d_;
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/slice4d_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/gradients/Slice_grad" />
declare const sliceGradConfig: GradConfig;

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/slice_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/slice_util" />
declare type SliceInfo = {
    finalShapeSparse: number[];
    finalShape: number[];
    isIdentity: boolean;
    sliceDim0: boolean;
    isSimpleSlice: boolean;
    begin: number[];
    end: number[];
    strides: number[];
};
declare function assertParamsValid(input: TensorInfo, begin: number[], size: number[]): void;
/** Converts a binary mask to an array of axes. Used in stridedSlice(). */
declare function maskToAxes(mask: number): number[];
/** Computes the output shape given the strided slice params. */
declare function computeOutShape(begin: number[], end: number[], strides: number[]): number[];
declare function stridesWithElidedDims(strides: number[], ellipsisInsertionIndex: number, numElidedAxes: number, inputShape: number[]): number[];
declare function getNormalizedAxes(inputShape: number[], ellipsisAxes: number[], numInterpolatedAxes: number, begin: number[], end: number[], strides: number[], beginMask: number, endMask: number, ellipsisMask: number): {
    begin: number[];
    end: number[];
    strides: number[];
};
declare function startIndicesWithElidedDims(beginMask: number, ellipsisInsertionIndex: number, numElidedAxes: number, originalBegin: number[], inputShape: number[]): number[];
declare function stopIndicesWithElidedDims(endMask: number, ellipsisInsertionIndex: number, numElidedAxes: number, originalEnd: number[], inputShape: number[]): number[];
declare function stridesForAxis(strides: number[], axis: number, ellipsisMask: number): number;
declare function startForAxis(beginMask: number, startIndices: number[], strides: number[], inputShape: number[], axis: number, ellipsisMask: number): number;
declare function stopForAxis(endMask: number, stopIndices: number[], strides: number[], inputShape: number[], axis: number, ellipsisMask: number): number;
/**
 * Returns true if the slice occupies a continous set of elements in the
 * 'flat' space.
 */
declare function isSliceContinous(shape: number[], begin: number[], size: number[]): boolean;
declare function computeFlatOffset(begin: number[], strides: number[]): number;
declare function parseSliceParams(x: TensorInfo, begin: number | number[], size?: number | number[]): number[][];
declare function sliceInfo(xShape: number[], begin: number[], end: number[], strides: number[], beginMask: number, endMask: number, ellipsisMask: number, newAxisMask: number, shrinkAxisMask: number): SliceInfo;

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/slice_util_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/public/chained_ops/softmax" />
declare module '../../tensor' {
    interface Tensor<R extends Rank = Rank> {
        softmax<T extends Tensor>(this: T, dim?: number): T;
    }
}
/// <amd-module name="@tensorflow/tfjs-core/dist/ops/losses/softmax_cross_entropy" />
/**
 * Computes the softmax cross entropy loss between two tensors.
 *
 * If labelSmoothing is nonzero, smooth the labels towards 1/2:
 *
 *   newOnehotLabels = onehotLabels * (1 - labelSmoothing)
 *                         + labelSmoothing / numClasses
 *
 * @param onehotLabels One hot encoded labels
 *    [batch_size, num_classes], same dimensions as 'predictions'.
 * @param logits The predicted outputs.
 * @param weights Tensor whose rank is either 0, or 1, and must be
 *    broadcastable to `loss`  of shape [batch_size]
 * @param labelSmoothing If greater than 0, then smooth the labels.
 * @param reduction Type of reduction to apply to loss. Should be of type
 *    `Reduction`
 *
 * @doc { heading: 'Training', subheading: 'Losses', namespace: 'losses' }
 */
declare function softmaxCrossEntropy_<T extends Tensor, O extends Tensor>(onehotLabels: T | TensorLike, logits: T | TensorLike, weights?: Tensor | TensorLike, labelSmoothing?: number, reduction?: Reduction): O;
declare const softmaxCrossEntropy: typeof softmaxCrossEntropy_;
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/losses/softmax_cross_entropy_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/gradients/Softmax_grad" />
declare const softmaxGradConfig: GradConfig;

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/softmax_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/public/chained_ops/softplus" />
declare module '../../tensor' {
    interface Tensor<R extends Rank = Rank> {
        softplus<T extends Tensor>(this: T): T;
    }
}

/// <amd-module name="@tensorflow/tfjs-core/dist/gradients/Softplus_grad" />
declare const softplusGradConfig: GradConfig;

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/softplus_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-data/dist/util/source_util" />
declare function isLocalPath(source: any): boolean;

/// <amd-module name="@tensorflow/tfjs-core/dist/gradients/SpaceToBatchND_grad" />
declare const spaceToBatchNDGradConfig: GradConfig;

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/space_to_batch_nd" />
/**
 * This operation divides "spatial" dimensions `[1, ..., M]` of the input into
 * a grid of blocks of shape `blockShape`, and interleaves these blocks with
 * the "batch" dimension (0) such that in the output, the spatial
 * dimensions `[1, ..., M]` correspond to the position within the grid,
 * and the batch dimension combines both the position within a spatial block
 * and the original batch position. Prior to division into blocks,
 * the spatial dimensions of the input are optionally zero padded
 * according to `paddings`. See below for a precise description.
 *
 * ```js
 * const x = tf.tensor4d([1, 2, 3, 4], [1, 2, 2, 1]);
 * const blockShape = [2, 2];
 * const paddings = [[0, 0], [0, 0]];
 *
 * x.spaceToBatchND(blockShape, paddings).print();
 * ```
 *
 * @param x A `tf.Tensor`. N-D with `x.shape` = `[batch] + spatialShape +
 * remainingShape`, where spatialShape has `M` dimensions.
 * @param blockShape A 1-D array. Must have shape `[M]`, all values must
 * be >= 1.
 * @param paddings A 2-D array. Must have shape `[M, 2]`, all values must be >=
 *     0. `paddings[i] = [padStart, padEnd]` specifies the amount to zero-pad
 * from input dimension `i + 1`, which corresponds to spatial dimension `i`. It
 * is required that
 * `(inputShape[i + 1] + padStart + padEnd) % blockShape[i] === 0`
 *
 * This operation is equivalent to the following steps:
 *
 * 1. Zero-pad the start and end of dimensions `[1, ..., M]` of the input
 * according to `paddings` to produce `padded` of shape paddedShape.
 *
 * 2. Reshape `padded` to `reshapedPadded` of shape:
 * `[batch] + [paddedShape[1] / blockShape[0], blockShape[0], ...,
 * paddedShape[M] / blockShape[M-1], blockShape[M-1]] + remainingShape`
 *
 * 3. Permute dimensions of `reshapedPadded` to produce `permutedReshapedPadded`
 * of shape: `blockShape + [batch] + [paddedShape[1] / blockShape[0], ...,
 * paddedShape[M] / blockShape[M-1]] + remainingShape`
 *
 * 4. Reshape `permutedReshapedPadded` to flatten `blockShape` into the
 * batch dimension, producing an output tensor of shape:
 * `[batch * prod(blockShape)] + [paddedShape[1] / blockShape[0], ...,
 * paddedShape[M] / blockShape[M-1]] + remainingShape`
 *
 * @doc {heading: 'Tensors', subheading: 'Transformations'}
 */
declare function spaceToBatchND_<T extends Tensor>(x: T | TensorLike, blockShape: number[], paddings: number[][]): T;
declare const spaceToBatchND: typeof spaceToBatchND_;
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/space_to_batch_nd_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/sparse/sparse_fill_empty_rows" />
/**
 * The input SparseTensor is represented via the map of inputs {`indices`,
 * `values`, `denseShape`}. The output SparseTensor has the same `denseShape`
 * but with indices `outputIndices` and values `outputValues`. This op inserts a
 * single entry for every row that doesn't have any values. The index is created
 * as `[row, 0, ..., 0]` and the inserted value is `defaultValue`.
 *
 * For example, suppose `spInput` has shape [5, 6] and non-empty values:
 * [0, 1]: a
 * [0, 3]: b
 * [2, 0]: c
 * [3, 1]: d
 *
 * Rows 1 and 4 are empty, so the output will be of shape [5, 6] with values:
 * [0, 1]: a
 * [0, 3]: b
 * [1, 0]: `defaultValue`
 * [2, 0]: c
 * [3, 1]: d
 * [4, 0]: `defaultValue`
 *
 * The output SparseTensor will be in row-major order and will have the same
 * shape as the input.
 *
 * This op also returns an indicator vector shaped [dense_shape[0]] such that
 * emptyRowIndicator[i] = True iff row i was an empty row.
 *
 * And a reverse index map vector shaped [indices.shape[0]] that is used during
 * backpropagation, reverseIndexMap[i] = outi s.t. indices[i, j] ==
 * outputIndices[outi, j] for all j
 *
 * ```js
 * const result = tf.sparse.sparseFillEmptyRows(
 *   [[0, 0], [1, 0], [1, 3], [1, 4], [3, 2], [3, 3]],
 *   [0, 10, 13, 14, 32, 33], [5, 6], -1);
 * console.log(result);
 * result['outputIndices'].print(); // [[0, 0], [1, 0], [1, 3], [1, 4],
 *                                  //  [2, 0], [3, 2], [3, 3], [4, 0]]
 * result['outputValues'].print(); // [0, 10, 13, 14,-1, 32, 33, -1]
 * result['emptyRowIndicator'].print(); // [false, false, true, false, true]
 * result['reverseIndexMap'].print(); // [0, 1, 2, 3, 5, 6]
 * ```
 * @param indices: 2-D. The indices of the sparse tensor.
 * @param values: 1-D. The values of the sparse tensor.
 * @param denseShape: 1-D. The shape of the sparse tensor.
 * @param defaultValue: 0-D. Default value to insert into location [row, 0, ...,
 *     0] for rows missing from the input sparse tensor.
 * @return A map with the following properties:
 *     - outputIndices
 *     - outputValues: 1-D. The values of the filled sparse tensor.
 *     - emptyRowIndicator: 1-D. Whether the dense row was missing in the input
 * sparse tensor.
 *     - reverseIndexMap: 1-D. A map from the input indices to the output
 * indices.
 * @doc {heading: 'Operations', subheading: 'Sparse'}
 */
declare function sparseFillEmptyRows_(indices: Tensor2D | TensorLike, values: Tensor1D | TensorLike, denseShape: Tensor1D | TensorLike, defaultValue: Scalar | ScalarLike): NamedTensorMap;
declare const sparseFillEmptyRows: typeof sparseFillEmptyRows_;
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/sparse/sparse_fill_empty_rows_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/sparse/sparse_fill_empty_rows_util" />
/**
 * Generates sparse fill empty rows indices, dense shape mismatch error message.
 *
 * @param indicesLength The first dimension of indices.
 */
declare function getSparseFillEmptyRowsIndicesDenseShapeMismatch(indicesLength: number): string;
/**
 * Generates sparse fill empty rows negative index error message.
 *
 * @param index The index with a negative value.
 * @param value The negative value.
 */
declare function getSparseFillEmptyRowsNegativeIndexErrorMessage(index: number, value: number): string;
/**
 * Generates sparse fill empty rows out of range index error message.
 *
 * @param index The index with an out of range value.
 * @param value The out of range value.
 * @param limit The upper limit for indices.
 */
declare function getSparseFillEmptyRowsOutOfRangeIndexErrorMessage(index: number, value: number, limit: number): string;

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/sparse/sparse_reshape" />
/**
 * This operation has the same semantics as reshape on the represented dense
 * tensor. The `inputIndices` are recomputed based on the requested `newShape`.
 * If one component of `newShape` is the special value -1, the size of that
 * dimension is computed so that the total dense size remains constant. At most
 * one component of `newShape` can be -1. The number of dense elements implied
 * by `newShape` must be the same as the number of dense elements originally
 * implied by `inputShape`. Reshaping does not affect the order of values in the
 * SparseTensor. If the input tensor has rank R_in and N non-empty values, and
 * `newShape` has length R_out, then `inputIndices` has shape [N, R_in],
 * `inputShape` has length R_in, `outputIndices` has shape [N, R_out], and
 * `outputShape` has length R_out.
 *
 * ```js
 * const result = tf.sparse.sparseReshape(
 *   [[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 2, 3]],
 *   [2, 3, 6], [9, -1]);
 * console.log(result);
 * result['outputIndices'].print(); //[[0, 0], [0, 1], [1, 2], [4, 2], [8, 1]]
 * result['outputShape'].print(); // [9, 4]
 * ```
 * @param inputIndices: 2-D. N x R_in matrix with the indices of non-empty
 * values in a SparseTensor.
 * @param inputShape: 1-D. R_in Tensor1D with the input SparseTensor's dense
 * shape.
 * @param newShape: 1-D. R_out Tensor1D with the requested new dense shape.
 * @return A map with the following properties:
 *     - outputIndices: 2-D. N x R_out matrix with the updated indices of
 *       non-empty values in the output SparseTensor.
 *     - outputShape: 1-D. R_out vector with the full dense shape of the output
 *       SparseTensor. This is the same as newShape but with any -1 dimensions
 *        filled in.
 * @doc {heading: 'Operations', subheading: 'Sparse'}
 */
declare function sparseReshape_(inputIndices: Tensor2D | TensorLike, inputShape: Tensor1D | TensorLike, newShape: Tensor1D | TensorLike): NamedTensorMap;
declare const sparseReshape: typeof sparseReshape_;
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/sparse/sparse_reshape_test" />
{ };
/// <amd-module name="@tensorflow/tfjs-core/dist/ops/sparse/sparse_reshape_util" />
/**
 * Generates sparse reshape multiple negative 1 output dimension error message.
 *
 * @param dim1 The first dimension with a negative 1 value.
 * @param dim2 The second dimension with a negative 1 value.
 */
declare function getSparseReshapeMultipleNegativeOneOutputDimErrorMessage(dim1: number, dim2: number): string;
/**
 * Generates sparse reshape negative output dimension error message.
 *
 * @param dim The dimension with a negative value.
 * @param value The negative value.
 */
declare function getSparseReshapeNegativeOutputDimErrorMessage(dim: number, value: number): string;
/**
 * Generates sparse reshape empty tensor zero output dimension error message.
 *
 */
declare function getSparseReshapeEmptyTensorZeroOutputDimErrorMessage(): string;
/**
 * Generates sparse reshape input output multiple mismatch error message.
 *
 * @param inputShape the input shape.
 * @param outputShape the requested output shape.
 */
declare function getSparseReshapeInputOutputMultipleErrorMessage(inputShape: number[], outputShape: number[]): string;
/**
 * Generates sparse reshape input output inequality error message.
 *
 * @param inputShape the input shape.
 * @param outputShape the requested output shape.
 */
declare function getSparseReshapeInputOutputMismatchErrorMessage(inputShape: number[], outputShape: number[]): string;

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/sparse/sparse_segment_mean" />
/**
 * Computes the mean along sparse segments of a tensor.
 *
 * ```js
 * const c = tf.tensor2d([[1,2,3,4], [-1,-2,-3,-4], [6,7,8,9]]);
 * // Select two rows, one segment.
 * const result1 = tf.sparse.sparseSegmentMean(c,
 *                                           tf.tensor1d([0, 1], 'int32'),
 *                                           tf.tensor1d([0, 0], 'int32'));
 * result1.print(); // [[0, 0, 0, 0]]
 *
 * // Select two rows, two segments.
 * const result2 = tf.sparse.sparseSegmentMean(c,
 *                                             tf.tensor1d([0, 1], 'int32'),
 *                                             tf.tensor1d([0, 1], 'int32'));
 * result2.print(); // [[1, 2, 3, 4], [-1, -2, -3, -4]]
 *
 * // Select all rows, two segments.
 * const result3 = tf.sparse.sparseSegmentMean(c,
 *                                             tf.tensor1d([0, 1, 2], 'int32'),
 *                                             tf.tensor1d([0, 1, 1], 'int32'));
 * result3.print(); // [[1.0, 2.0, 3.0, 4.0], [2.5, 2.5, 2.5, 2.5]]
 * ```
 * @param data: A Tensor of at least one dimension with data that will be
 *     assembled in the output.
 * @param indices: A 1-D Tensor with indices into data. Has same rank as
 *     segmentIds.
 * @param segmentIds: A 1-D Tensor with indices into the output Tensor. Values
 *     should be sorted and can be repeated.
 * @return Has same shape as data, except for dimension 0 which has equal to
 *         the number of segments.
 *
 * @doc {heading: 'Operations', subheading: 'Sparse'}
 */
declare function sparseSegmentMean_(data: Tensor | TensorLike, indices: Tensor1D | TensorLike, segmentIds: Tensor1D | TensorLike): Tensor;
declare const sparseSegmentMean: typeof sparseSegmentMean_;
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/sparse/sparse_segment_mean_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/sparse/sparse_segment_reduction_util" />
/**
 * Generates sparse segment reduction negative segment ids error message.
 *
 */
declare function getSparseSegmentReductionNegativeSegmentIdsErrorMessage(): string;
/**
 * Generates sparse segment reduction non increasing segment ids error message.
 *
 */
declare function getSparseSegmentReductionNonIncreasingSegmentIdsErrorMessage(): string;
/**
 * Generates sparse segment reduction segment id out of range error message.
 *
 * @param segmentId The segment id index that is out of range.
 * @param outputRows Upper bound of valid segment id values.
 */
declare function getSparseSegmentReductionSegmentIdOutOfRangeErrorMessage(segmentId: number, outputRows: number): string;
/**
 * Generates sparse segment reduction input indice out of range error message.
 *
 * @param index The index that holds the out of range value.
 * @param indexValue The value that is out of range.
 * @param inputRows Upper bound of valid index values.
 */
declare function getSparseSegmentReductionIndicesOutOfRangeErrorMessage(index: number, indexValue: number, inputRows: number): string;

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/sparse/sparse_segment_sum" />
/**
 * Computes the sum along sparse segments of a tensor.
 *
 * ```js
 * const c = tf.tensor2d([[1,2,3,4], [-1,-2,-3,-4], [5,6,7,8]]);
 * // Select two rows, one segment.
 * const result1 = tf.sparse.sparseSegmentSum(c,
 *                                           tf.tensor1d([0, 1], 'int32'),
 *                                           tf.tensor1d([0, 0], 'int32'));
 * result1.print(); // [[0, 0, 0, 0]]
 *
 * // Select two rows, two segments.
 * const result2 = tf.sparse.sparseSegmentSum(c,
 *                                           tf.tensor1d([0, 1], 'int32'),
 *                                           tf.tensor1d([0, 1], 'int32'));
 * result2.print(); // [[1, 2, 3, 4], [-1, -2, -3, -4]]
 *
 * // Select all rows, two segments.
 * const result3 = tf.sparse.sparseSegmentSum(c,
 *                                           tf.tensor1d([0, 1, 2], 'int32'),
 *                                           tf.tensor1d([0, 0, 1], 'int32'));
 * result3.print(); // [[0, 0, 0, 0], [5, 6, 7, 8]]
 * ```
 * @param data: A Tensor of at least one dimension with data that will be
 *     assembled in the output.
 * @param indices: A 1-D Tensor with indices into data. Has same rank as
 *     segmentIds.
 * @param segmentIds: A 1-D Tensor with indices into the output Tensor. Values
 *     should be sorted and can be repeated.
 * @return Has same shape as data, except for dimension 0 which has equal to
 *         the number of segments.
 *
 * @doc {heading: 'Operations', subheading: 'Sparse'}
 */
declare function sparseSegmentSum_(data: Tensor | TensorLike, indices: Tensor1D | TensorLike, segmentIds: Tensor1D | TensorLike): Tensor;
declare const sparseSegmentSum: typeof sparseSegmentSum_;
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/sparse/sparse_segment_sum_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/sparse_to_dense" />
/**
 * Converts a sparse representation into a dense tensor.
 *
 * Builds an array dense with shape outputShape such that:
 *
 * // If sparseIndices is scalar
 * dense[i] = (i == sparseIndices ? sparseValues : defaultValue)
 *
 * // If sparseIndices is a vector, then for each i
 * dense[sparseIndices[i]] = sparseValues[i]
 *
 * // If sparseIndices is an n by d matrix, then for each i in [0, n)
 * dense[sparseIndices[i][0], ..., sparseIndices[i][d-1]] = sparseValues[i]
 * All other values in dense are set to defaultValue. If sparseValues is a
 * scalar, all sparse indices are set to this single value.
 *
 * If indices are repeated the final value is summed over all values for those
 * indices.
 *
 * ```js
 * const indices = tf.tensor1d([4, 5, 6, 1, 2, 3], 'int32');
 * const values = tf.tensor1d([10, 11, 12, 13, 14, 15], 'float32');
 * const shape = [8];
 * tf.sparseToDense(indices, values, shape).print();
 * ```
 *
 * @param sparseIndices A 0-D, 1-D, or 2-D Tensor of type int32.
 * sparseIndices[i] contains the complete index where sparseValues[i] will be
 * placed.
 * @param sparseValues A 0-D or 1-D Tensor. Values
 * corresponding to each row of sparseIndices, or a scalar value to be used for
 * all sparse indices.
 * @param outputShape Shape of the dense output tensor. The type is inferred.
 * @param defaultValue Scalar. Value to set for indices not specified in
 * sparseIndices. Defaults to zero.
 *
 * @doc {heading: 'Operations', subheading: 'Normalization'}
 */
declare function sparseToDense_<R extends Rank>(sparseIndices: Tensor | TensorLike, sparseValues: Tensor | TensorLike, outputShape: ShapeMap[R], defaultValue?: Scalar | ScalarLike): Tensor<R>;
declare const sparseToDense: typeof sparseToDense_;
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/sparse_to_dense_test" />
{ };
/// <amd-module name="@tensorflow/tfjs-core/dist/ops/sparse_to_dense_util" />

/**
 * Validate sparseToDense inputs.
 *
 * @param sparseIndices A 0-D, 1-D, or 2-D Tensor of type int32.
 * sparseIndices[i] contains the complete index where sparseValues[i] will be
 * placed.
 * @param sparseValues A 0-D or 1-D Tensor. Values
 * corresponding to each row of sparseIndices, or a scalar value to be used for
 * all sparse indices.
 * @param outputShape number[]. Shape of the dense output tensor.
 * @param validateIndices boolean. indice validation is not supported, error
 * will be thrown if it is set.
 */
declare function validateInput(sparseIndices: Tensor, sparseValues: Tensor, outputShape: number[], defaultValues: Tensor): void;
/// <amd-module name="@tensorflow/tfjs-core/dist/ops/split" />
/**
 * Splits a `tf.Tensor` into sub tensors.
 *
 * If `numOrSizeSplits` is a number, splits `x` along dimension `axis`
 * into `numOrSizeSplits` smaller tensors.
 * Requires that `numOrSizeSplits` evenly divides `x.shape[axis]`.
 *
 * If `numOrSizeSplits` is a number array, splits `x` into
 * `numOrSizeSplits.length` pieces. The shape of the `i`-th piece has the
 * same size as `x` except along dimension `axis` where the size is
 * `numOrSizeSplits[i]`.
 *
 * ```js
 * const x = tf.tensor2d([1, 2, 3, 4, 5, 6, 7, 8], [2, 4]);
 * const [a, b] = tf.split(x, 2, 1);
 * a.print();
 * b.print();
 *
 * const [c, d, e] = tf.split(x, [1, 2, 1], 1);
 * c.print();
 * d.print();
 * e.print();
 * ```
 *
 * @param x The input tensor to split.
 * @param numOrSizeSplits Either an integer indicating the number of
 * splits along the axis or an array of integers containing the sizes of
 * each output tensor along the axis. If a number then it must evenly divide
 * `x.shape[axis]`; otherwise the sum of sizes must match `x.shape[axis]`.
 * Can contain one -1 indicating that dimension is to be inferred.
 * @param axis The dimension along which to split. Defaults to 0 (the first
 * dim).
 *
 * @doc {heading: 'Tensors', subheading: 'Slicing and Joining'}
 */
declare function split_<T extends Tensor>(x: Tensor | TensorLike, numOrSizeSplits: number[] | number, axis?: number): T[];
declare const split: typeof split_;
{ };
/// <amd-module name="@tensorflow/tfjs-core/dist/gradients/SplitV_grad" />
declare const splitVGradConfig: GradConfig;

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/split_test" />
{ };
/// <amd-module name="@tensorflow/tfjs-core/dist/ops/split_util" />

/**
 * Prepare the split size array. When the input is a number, the axis is evenly
 * divided among the split size. When the input contains the negative value, the
 * rest of the axis is allocated toward that.
 */
declare function prepareSplitSize(x: Tensor | TensorInfo, numOrSizeSplits: number[] | number, axis?: number): number[];

/// <amd-module name="@tensorflow/tfjs-core/dist/public/chained_ops/sqrt" />
declare module '../../tensor' {
    interface Tensor<R extends Rank = Rank> {
        sqrt<T extends Tensor>(this: T): T;
    }
}

/// <amd-module name="@tensorflow/tfjs-core/dist/gradients/Sqrt_grad" />
declare const sqrtGradConfig: GradConfig;

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/sqrt_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/square" />
/**
 * Computes square of `x` element-wise: `x ^ 2`
 *
 * ```js
 * const x = tf.tensor1d([1, 2, Math.sqrt(2), -1]);
 *
 * x.square().print();  // or tf.square(x)
 * ```
 * @param x The input Tensor.
 *
 * @doc {heading: 'Operations', subheading: 'Basic math'}
 */
declare function square_<T extends Tensor>(x: T | TensorLike): T;
declare const square: typeof square_;
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/gradients/SquaredDifference_grad" />
declare const squaredDifferenceGradConfig: GradConfig;

/// <amd-module name="@tensorflow/tfjs-core/dist/public/chained_ops/squared_difference" />
declare module '../../tensor' {
    interface Tensor<R extends Rank = Rank> {
        squaredDifference<T extends Tensor>(b: Tensor | TensorLike): T;
    }
}

/// <amd-module name="@tensorflow/tfjs-core/dist/gradients/Square_grad" />
declare const squareGradConfig: GradConfig;

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/square_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/squeeze" />
/**
 * Removes dimensions of size 1 from the shape of a `tf.Tensor`.
 *
 * ```js
 * const x = tf.tensor([1, 2, 3, 4], [1, 1, 4]);
 * x.squeeze().print();
 * ```
 *
 * @param x The input tensor to be squeezed.
 * @param axis An optional list of numbers. If specified, only
 *     squeezes the dimensions listed. The dimension index starts at 0. It
 * is an error to squeeze a dimension that is not 1.
 *
 * @doc {heading: 'Tensors', subheading: 'Transformations'}
 */
declare function squeeze_<T extends Tensor>(x: Tensor | TensorLike, axis?: number[]): T;
declare const squeeze: typeof squeeze_;
{ };
/// <amd-module name="@tensorflow/tfjs-core/dist/ops/squeeze_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/stack" />
/**
 * Stacks a list of rank-`R` `tf.Tensor`s into one rank-`(R+1)` `tf.Tensor`.
 *
 * ```js
 * const a = tf.tensor1d([1, 2]);
 * const b = tf.tensor1d([3, 4]);
 * const c = tf.tensor1d([5, 6]);
 * tf.stack([a, b, c]).print();
 * ```
 *
 * @param tensors A list of tensor objects with the same shape and dtype.
 * @param axis The axis to stack along. Defaults to 0 (the first dim).
 *
 * @doc {heading: 'Tensors', subheading: 'Slicing and Joining'}
 */
declare function stack_<T extends Tensor>(tensors: Array<T | TensorLike>, axis?: number): Tensor;
declare const stack: typeof stack_;
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/stack_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-layers/dist/backend/state" />
declare function getNextUniqueTensorId(): number;
/**
 * Provides a unique UID given a string prefix.
 *
 * @param prefix
 */
declare function getUid(prefix?: string): string;

/// <amd-module name="@tensorflow/tfjs-data/dist/statistics" />
/**
 * The value associated with a given key for a single element.
 *
 * Such a value may not have a batch dimension.  A value may be a scalar or an
 * n-dimensional array.
 */
declare type ElementArray = number | number[] | tf.Tensor | string;
/**
 * A map from string keys (aka column names) to values for a single element.
 */
declare type TabularRecord = {
    [key: string]: ElementArray;
};
/** An interface representing numeric statistics of a column. */
interface NumericColumnStatistics {
    min: number;
    max: number;
    mean: number;
    variance: number;
    stddev: number;
    length: number;
}
/**
 * An interface representing column level NumericColumnStatistics for a
 * Dataset.
 */
interface DatasetStatistics {
    [key: string]: NumericColumnStatistics;
}
/**
 * Provides a function that scales numeric values into the [0, 1] interval.
 *
 * @param min the lower bound of the inputs, which should be mapped to 0.
 * @param max the upper bound of the inputs, which should be mapped to 1,
 * @return A function that maps an input ElementArray to a scaled ElementArray.
 */
declare function scaleTo01(min: number, max: number): (value: ElementArray) => ElementArray;
/**
 * Provides a function that calculates column level statistics, i.e. min, max,
 * variance, stddev.
 *
 * @param dataset The Dataset object whose statistics will be calculated.
 * @param sampleSize (Optional) If set, statistics will only be calculated
 *     against a subset of the whole data.
 * @param shuffleWindowSize (Optional) If set, shuffle provided dataset before
 *     calculating statistics.
 * @return A DatasetStatistics object that contains NumericColumnStatistics of
 *     each column.
 */
declare function computeDatasetStatistics(dataset: Dataset<TabularRecord>, sampleSize?: number, shuffleWindowSize?: number): Promise<DatasetStatistics>;

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/step" />
/**
 * Computes step of the input `tf.Tensor` element-wise: `x > 0 ? 1 : alpha`
 *
 * ```js
 * const x = tf.tensor1d([0, 2, -1, -3]);
 *
 * x.step(.5).print();  // or tf.step(x, .5)
 * ```
 * @param x The input tensor.
 * @param alpha The gradient when input is negative. Defaults to 0.
 *
 * @doc {heading: 'Operations', subheading: 'Basic math'}
 */
declare function step_<T extends Tensor>(x: T | TensorLike, alpha?: number): T;
declare const step: typeof step_;
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/gradients/Step_grad" />
declare const stepGradConfig: GradConfig;

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/step_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/signal/stft" />
/**
 * Computes the Short-time Fourier Transform of signals
 * See: https://en.wikipedia.org/wiki/Short-time_Fourier_transform
 *
 * ```js
 * const input = tf.tensor1d([1, 1, 1, 1, 1])
 * tf.signal.stft(input, 3, 1).print();
 * ```
 * @param signal 1-dimensional real value tensor.
 * @param frameLength The window length of samples.
 * @param frameStep The number of samples to step.
 * @param fftLength The size of the FFT to apply.
 * @param windowFn A callable that takes a window length and returns 1-d tensor.
 *
 * @doc {heading: 'Operations', subheading: 'Signal', namespace: 'signal'}
 */
declare function stft_(signal: Tensor1D, frameLength: number, frameStep: number, fftLength?: number, windowFn?: (length: number) => Tensor1D): Tensor;
declare const stft: typeof stft_;
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/signal/stft_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/strided_slice" />
/**
 * Extracts a strided slice of a tensor.
 *
 * Roughly speaking, this op extracts a slice of size (end-begin)/stride from
 * the given input tensor (x). Starting at the location specified by begin the
 * slice continues by adding stride to the index until all dimensions are not
 * less than end. Note that a stride can be negative, which causes a reverse
 * slice.
 *
 * ```js
 * const t = tf.tensor3d([1, 1, 1 ,2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6],
 *    [3, 2, 3]);
 * t.stridedSlice([1, 0, 0], [2, 1, 3], [1, 1, 1]).print()  // [[[3, 3, 3]]]
 * t.stridedSlice([1, 0, 0], [2, 2, 3], [1, 1, 1]).print()  // [[[3, 3, 3],
 *                                                     // [4, 4, 4]]]
 * t.stridedSlice([1, -1, 0], [2, -3, 3], [1, -1, 1]).print() // [[[4, 4, 4],
 *                                                     // [3, 3, 3]]]
 * ```
 *
 * @param x The tensor to stride slice.
 * @param begin The coordinates to start the slice from.
 * @param end: The coordinates to end the slice at.
 * @param strides: The size of the slice.
 * @param beginMask: If the ith bit of beginMask is set, begin[i] is ignored
 *      and the fullest possible range in that dimension is used instead.
 * @param endMask: If the ith bit of endMask is set, end[i] is ignored
 *      and the fullest possible range in that dimension is used instead.
 * @param shrinkAxisMask: a bitmask where bit i implies that
 * the ith specification should shrink the dimensionality. begin and end must
 * imply a slice of size 1 in the dimension.
 *
 * @doc {heading: 'Operations', subheading: 'Slicing and Joining'}
 */
declare function stridedSlice_(x: Tensor | TensorLike, begin: number[], end: number[], strides?: number[], beginMask?: number, endMask?: number, ellipsisMask?: number, newAxisMask?: number, shrinkAxisMask?: number): Tensor;
declare const stridedSlice: typeof stridedSlice_;
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/strided_slice_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-data/dist/iterators/string_iterator" />
declare abstract class StringIterator extends LazyIterator<string> {
    /**
     * Splits a string stream on a given separator.
     *
     * It is assumed that the incoming chunk boundaries have no semantic meaning,
     * so conceptually the incoming stream is treated simply as the concatenation
     * of its elements.
     *
     * The outgoing stream provides chunks corresponding to the results of the
     * standard string split() operation (even if such a chunk spanned incoming
     * chunks).  The separators are not included.
     *
     * A typical usage is to split a text file (represented as a stream with
     * arbitrary chunk boundaries) into lines.
     *
     * @param upstream A readable stream of strings that can be treated as
     *   concatenated.
     * @param separator A character to split on.
     */
    split(separator: string): StringIterator;
}

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/string/string_n_grams" />
/**
 * Creates ngrams from ragged string data.
 *
 * This op accepts a ragged tensor with 1 ragged dimension containing only
 * strings and outputs a ragged tensor with 1 ragged dimension containing ngrams
 * of that string, joined along the innermost axis.
 *
 * ```js
 * const result = tf.string.stringNGrams(
 *   ['a', 'b', 'c', 'd'], tf.tensor1d([0, 2, 4], 'int32'),
 *   '|', [1, 2], 'LP', 'RP', -1, false);
 * result['nGrams'].print(); // ['a', 'b', 'LP|a', 'a|b', 'b|RP',
 *                           //  'c', 'd', 'LP|c', 'c|d', 'd|RP']
 * result['nGramsSplits'].print(); // [0, 5, 10]
 * ```
 * @param data: The values tensor of the ragged string tensor to make ngrams out
 *     of. Must be a 1D string tensor.
 * @param dataSplits: The splits tensor of the ragged string tensor to make
 *     ngrams out of.
 * @param separator: The string to append between elements of the token. Use ""
 *     for no separator.
 * @param nGramWidths: The sizes of the ngrams to create.
 * @param leftPad: The string to use to pad the left side of the ngram sequence.
 *     Only used if pad_width !== 0.
 * @param rightPad: The string to use to pad the right side of the ngram
 *     sequence. Only used if pad_width !== 0.
 * @param padWidth: The number of padding elements to add to each side of each
 *     sequence. Note that padding will never be greater than `nGramWidths`-1
 *     regardless of this value. If `padWidth`=-1, then add max(`nGramWidths`)-1
 *     elements.
 * @param preserveShortSequences: If true, then ensure that at least one ngram
 *     is generated for each input sequence. In particular, if an input sequence
 *     is shorter than min(ngramWidth) + 2*padWidth, then generate a single
 *     ngram containing the entire sequence. If false, then no ngrams are
 *     generated for these short input sequences.
 * @return A map with the following properties:
 *     - nGrams: The values tensor of the output ngrams ragged tensor.
 *     - nGramsSplits: The splits tensor of the output ngrams ragged tensor.
 *
 * @doc {heading: 'Operations', subheading: 'String'}
 */
declare function stringNGrams_(data: Tensor1D | TensorLike, dataSplits: Tensor | TensorLike, separator: string, nGramWidths: number[], leftPad: string, rightPad: string, padWidth: number, preserveShortSequences: boolean): NamedTensorMap;
declare const stringNGrams: typeof stringNGrams_;
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/string/string_n_grams_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/string/string_split" />
/**
 * Split elements of `input` based on `delimiter` into a SparseTensor .
 *
 * Let N be the size of source (typically N will be the batch size). Split each
 * element of `input` based on `delimiter` and return a SparseTensor containing
 * the splitted tokens. Empty tokens are ignored if `skipEmpty` is set to True.
 *
 * `delimiter` can be empty, or a string of split characters. If `delimiter` is
 * an empty string, each element of `input` is split into individual
 * character strings. Otherwise every character of `delimiter` is a potential
 * split point.
 *
 * ```js
 * const result = tf.string.stringSplit(['hello world',  'a b c'], ' ');
 * result['indices'].print(); // [[0, 0], [0, 1], [1, 0], [1, 1], [1, 2]]
 * result['values'].print(); // ['hello', 'world', 'a', 'b', 'c']
 * result['shape'].print(); // [2, 3]
 * ```
 * @param input: 1-D. Strings to split.
 * @param delimiter: 0-D. Delimiter characters, or empty string.
 * @param skipEmpty: Optional. If true, skip the empty strings from the result.
 *     Defaults to true.
 * @return A map with the following properties:
 *     - indices: A dense matrix of int32 representing the indices of the sparse
 *       tensor.
 *     - values: A vector of strings corresponding to the splited values.
 *     - shape: a length-2 vector of int32 representing the shape of the sparse
 * tensor, where the first value is N and the second value is the maximum number
 * of tokens in a single input entry.
 *
 * @doc {heading: 'Operations', subheading: 'String'}
 */
declare function stringSplit_(input: Tensor1D | TensorLike, delimiter: Scalar | ScalarLike, skipEmpty?: boolean): NamedTensorMap;
declare const stringSplit: typeof stringSplit_;
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/string/string_split_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/string/string_to_hash_bucket_fast" />
/**
 * Converts each string in the input Tensor to its hash mod by a number of
 * buckets.
 *
 * The hash function is deterministic on the content of the string within the
 * process and will never change. However, it is not suitable for cryptography.
 * This function may be used when CPU time is scarce and inputs are trusted or
 * unimportant. There is a risk of adversaries constructing inputs that all hash
 * to the same bucket.
 *
 * ```js
 * const result = tf.string.stringToHashBucketFast(
 *   ['Hello', 'TensorFlow', '2.x'], 3);
 * result.print(); // [0, 2, 2]
 * ```
 * @param input: The strings to assign a hash bucket.
 * @param numBuckets: The number of buckets.
 * @return A Tensor of the same shape as the input tensor.
 *
 * @doc {heading: 'Operations', subheading: 'String'}
 */
declare function stringToHashBucketFast_(input: Tensor | TensorLike, numBuckets: number): Tensor;
declare const stringToHashBucketFast: typeof stringToHashBucketFast_;
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/string/string_to_hash_bucket_fast_test" />
{ };
/// <amd-module name="@tensorflow/tfjs-core/dist/public/chained_ops/sub" />
declare module '../../tensor' {
    interface Tensor<R extends Rank = Rank> {
        sub<T extends Tensor>(b: Tensor | TensorLike): T;
    }
}
/// <amd-module name="@tensorflow/tfjs-core/dist/gradients/Sub_grad" />
declare const subGradConfig: GradConfig;
/// <amd-module name="@tensorflow/tfjs-core/dist/ops/sub_test" />
{ };
/// <amd-module name="@tensorflow/tfjs-core/dist/ops/sum" />
/**
 * Computes the sum of elements across dimensions of a `tf.Tensor`.
 *
 * Reduces the input along the dimensions given in `axes`. Unless `keepDims`
 * is true, the rank of the `tf.Tensor` is reduced by 1 for each entry in
 * `axes`. If `keepDims` is true, the reduced dimensions are retained with
 * length 1. If axes has no entries, all dimensions are reduced, and a
 * `tf.Tensor` with a single element is returned.
 *
 * ```js
 * const x = tf.tensor1d([1, 2, 3]);
 *
 * x.sum().print();  // or tf.sum(x)
 * ```
 *
 * ```js
 * const x = tf.tensor2d([1, 2, 3, 4], [2, 2]);
 *
 * const axis = 1;
 * x.sum(axis).print();  // or tf.sum(x, axis)
 * ```
 *
 * @param x The input tensor to compute the sum over. If the dtype is `bool`
 *   it will be converted to `int32` and the output dtype will be `int32`.
 * @param axis The dimension(s) to reduce. By default it reduces
 *     all dimensions.
 * @param keepDims If true, retains reduced dimensions with size 1.
 *
 * @doc {heading: 'Operations', subheading: 'Reduction'}
 */
declare function sum_<T extends Tensor>(x: Tensor | TensorLike, axis?: number | number[], keepDims?: boolean): T;
declare const sum: typeof sum_;
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/gradients/Sum_grad" />
declare const sumGradConfig: GradConfig;

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/sum_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/public/chained_ops/tan" />
declare module '../../tensor' {
    interface Tensor<R extends Rank = Rank> {
        tan<T extends Tensor>(this: T): T;
    }
}

/// <amd-module name="@tensorflow/tfjs-core/dist/public/chained_ops/tanh" />
declare module '../../tensor' {
    interface Tensor<R extends Rank = Rank> {
        tanh<T extends Tensor>(this: T): T;
    }
}

/// <amd-module name="@tensorflow/tfjs-core/dist/gradients/Tanh_grad" />
declare const tanhGradConfig: GradConfig;

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/tanh_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/gradients/Tan_grad" />
declare const tanGradConfig: GradConfig;

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/tan_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/tape" />
interface TapeNode {
    id: number;
    kernelName: string;
    outputs: Tensor[];
    inputs: NamedTensorMap;
    gradient?: (dys: Tensor[]) => NamedGradientMap;
    saved?: Tensor[];
}
declare type NamedGradientMap = {
    [inputName: string]: () => Tensor;
};
/**
 * Computes a list of TapeNodes that connect x to y, filtering everything else
 * out and preserving the order of the original tape elements.
 *
 * @param tape The tape elements to filter.
 * @param xs The input Tensors.
 * @param y The output Tensor.
 */
declare function getFilteredNodesXToY(tape: TapeNode[], xs: Tensor[], y: Tensor): TapeNode[];
/**
 * Backpropagate gradients through the filtered TapeNodes.
 *
 * @param tensorAccumulatedGradientMap A map of Tensor to its gradient. This map
 * is mutated by this method.
 * @param filteredTape The filtered TapeNodes to backprop through.
 */
declare function backpropagateGradients(tensorAccumulatedGradientMap: {
    [tensorId: number]: Tensor;
}, filteredTape: TapeNode[], tidy: (f: Function) => Tensor, add: (a: Tensor, b: Tensor) => Tensor): void;

/// <amd-module name="@tensorflow/tfjs-core/dist/tape_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/tensor" />
/**
 * Creates a `tf.Tensor` with the provided values, shape and dtype.
 *
 * ```js
 * // Pass an array of values to create a vector.
 * tf.tensor([1, 2, 3, 4]).print();
 * ```
 *
 * ```js
 * // Pass a nested array of values to make a matrix or a higher
 * // dimensional tensor.
 * tf.tensor([[1, 2], [3, 4]]).print();
 * ```
 *
 * ```js
 * // Pass a flat array and specify a shape yourself.
 * tf.tensor([1, 2, 3, 4], [2, 2]).print();
 * ```
 *
 * ```js
 * // Pass a `WebGLData` object and specify a shape yourself.
 *
 * // This makes it possible for TF.js applications to avoid GPU / CPU sync.
 * // For example, if your application includes a preprocessing step on the GPU,
 * // you could upload the GPU output directly to TF.js, rather than first
 * // downloading the values.
 *
 * // Example for WebGL2:
 * const customCanvas = document.createElement('canvas');
 * const customBackend = new tf.MathBackendWebGL(customCanvas);
 * tf.registerBackend('custom-webgl', () => customBackend);
 * await tf.setBackend('custom-webgl');
 * const gl = customBackend.gpgpu.gl;
 * const texture = gl.createTexture();
 * const tex2d = gl.TEXTURE_2D;
 * const width = 2;
 * const height = 2;
 *
 * gl.bindTexture(tex2d, texture);
 * gl.texParameteri(tex2d, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
 * gl.texParameteri(tex2d, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
 * gl.texParameteri(tex2d, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
 * gl.texParameteri(tex2d, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
 * gl.texImage2D(
 *   tex2d, 0, gl.RGBA32F, // internalFormat
 *   width, height, 0,
 *   gl.RGBA, // textureFormat
 *   gl.FLOAT, // textureType
 *   new Float32Array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
 * );
 *
 * // Currently, the `texture` has 4 pixels:
 * // Pixel0 is {R:0, G:1, B:2, A:3}
 * // Pixel1 is {R:4, G:5, B:6, A:7}
 * // Pixel2 is {R:8, G:9, B:10, A:11}
 * // Pixel3 is {R:12, G:13, B:14, A:15}
 *
 * const logicalShape = [height * width * 2];
 * const a = tf.tensor({texture, height, width, channels: 'BR'}, logicalShape);
 * // Tensor value will be [2, 0, 6, 4, 10, 8, 14, 12], since [2, 0] is the
 * // values of 'B' and 'R' channels of Pixel0, [6, 4] is the values of 'B' and
 * 'R'
 * // channels of Pixel1...
 *
 * // For postprocessing on the GPU, it's possible to retrieve the texture
 * // backing any tensor by calling the tensor's `dataToGPU` method like
 * // so:
 *
 * const tex = a.dataToGPU();
 * ```
 *
 * ```js
 * // Pass a `WebGPUData` object and specify a shape yourself.
 *
 * // This makes it possible for TF.js applications to avoid GPU / CPU sync.
 * // For example, if your application includes a preprocessing step on the GPU,
 * // you could upload the GPU output directly to TF.js, rather than first
 * // downloading the values. Unlike WebGL, this optionally supports zero copy
 * // by WebGPUData.zeroCopy. When zeroCopy is false or undefined(default), this
 * // passing GPUBuffer can be destroyed after tensor is created. When zeroCopy
 * // is true, this GPUBuffer is bound directly by the tensor, so do not destroy
 * // this GPUBuffer until all access is done.
 *
 * // Example for WebGPU:
 * function createGPUBufferFromData(device, data, dtype) {
 *   const bytesPerElement = 4;
 *   const sizeInBytes = data.length * bytesPerElement;
 *
 *   const gpuWriteBuffer = device.createBuffer({
 *     mappedAtCreation: true,
 *     size: sizeInBytes,
 *     usage: GPUBufferUsage.MAP_WRITE | GPUBufferUsage.COPY_SRC
 *   });
 *   const arrayBuffer = gpuWriteBuffer.getMappedRange();
 *   if (dtype === 'float32') {
 *     new Float32Array(arrayBuffer).set(data);
 *   } else if (dtype === 'int32') {
 *     new Int32Array(arrayBuffer).set(data);
 *   } else {
 *     throw new Error(
 *         `Creating tensor from GPUBuffer only supports` +
 *         `'float32'|'int32' dtype, while the dtype is ${dtype}.`);
 *   }
 *   gpuWriteBuffer.unmap();
 *
 *   const gpuReadBuffer = device.createBuffer({
 *     mappedAtCreation: false,
 *     size: sizeInBytes,
 *     usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE |
 *         GPUBufferUsage.COPY_SRC
 *   });
 *
 *   const copyEncoder = device.createCommandEncoder();
 *   copyEncoder.copyBufferToBuffer(
 *       gpuWriteBuffer, 0, gpuReadBuffer, 0, sizeInBytes);
 *   const copyCommands = copyEncoder.finish();
 *   device.queue.submit([copyCommands]);
 *   gpuWriteBuffer.destroy();
 *   return gpuReadBuffer;
 * }
 *
 * const dtype = 'float32';
 * const device = tf.backend().device;
 * const aData = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
 * const bData = [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4];
 * const expected = [2, 4, 6, 8, 6, 8, 10, 12, 10, 12, 14, 16, 14, 16, 18, 20];
 * const aBuffer = createGPUBufferFromData(device, aData, dtype);
 * const shape = [aData.length];
 * // To use zeroCopy, use {buffer: aBuffer, zeroCopy: true} instead and destroy
 * // aBuffer untill all access is done.
 * const a = tf.tensor({buffer: aBuffer}, shape, dtype);
 * const b = tf.tensor(bData, shape, dtype);
 * const result = tf.add(a, b);
 * a.dispose();
 * b.dispose();
 * result.dispose();
 * aBuffer.destroy();
 * ```
 * @param values The values of the tensor. Can be nested array of numbers,
 *     or a flat array, or a `TypedArray`, or a `WebGLData` object, or a
 * `WebGPUData` object. If the values are strings, they will be encoded as utf-8
 * and kept as `Uint8Array[]`. If the values is a `WebGLData` object, the dtype
 * could only be 'float32' or 'int32' and the object has to have: 1. texture, a
 * `WebGLTexture`, the texture must share the same `WebGLRenderingContext` with
 * TFJS's WebGL backend (you could create a custom WebGL backend from your
 * texture's canvas) and the internal texture format for the input texture must
 * be floating point or normalized integer; 2. height, the height of the
 * texture; 3. width, the width of the texture; 4. channels, a non-empty subset
 * of 'RGBA', indicating the values of which channels will be passed to the
 * tensor, such as 'R' or 'BR' (The order of the channels affect the order of
 * tensor values. ). (If the values passed from texture is less than the tensor
 * size, zeros will be padded at the rear.). If the values is a `WebGPUData`
 * object, the dtype could only be 'float32' or 'int32 and the object has to
 * have: buffer, a `GPUBuffer`. The buffer must: 1. share the same `GPUDevice`
 * with TFJS's WebGPU backend; 2. buffer.usage should at least support
 * GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC; 3. buffer.size should not
 * be smaller than the byte size of tensor shape. WebGPUData optionally supports
 * zero copy by flag zeroCopy. When zeroCopy is false or undefined(default),
 * this passing GPUBuffer can be destroyed after tensor is created. When
 * zeroCopy is true, this GPUBuffer is bound directly by the tensor, so do not
 * destroy this GPUBuffer until all access is done.
 * @param shape The shape of the tensor. Optional. If not provided,
 *   it is inferred from `values`.
 * @param dtype The data type.
 *
 * @doc {heading: 'Tensors', subheading: 'Creation'}
 */
declare function tensor<R extends Rank>(values: TensorLike | WebGLData | WebGPUData, shape?: ShapeMap[R], dtype?: DataType): Tensor<R>;

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/tensor1d" />
/**
 * Creates rank-1 `tf.Tensor` with the provided values, shape and dtype.
 *
 * The same functionality can be achieved with `tf.tensor`, but in general
 * we recommend using `tf.tensor1d` as it makes the code more readable.
 *
 * ```js
 * tf.tensor1d([1, 2, 3]).print();
 * ```
 *
 * @param values The values of the tensor. Can be array of numbers,
 *     or a `TypedArray`.
 * @param dtype The data type.
 *
 * @doc {heading: 'Tensors', subheading: 'Creation'}
 */
declare function tensor1d(values: TensorLike1D, dtype?: DataType): Tensor1D;

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/tensor2d" />
/**
 * Creates rank-2 `tf.Tensor` with the provided values, shape and dtype.
 *
 * The same functionality can be achieved with `tf.tensor`, but in general
 * we recommend using `tf.tensor2d` as it makes the code more readable.
 *
 *  ```js
 * // Pass a nested array.
 * tf.tensor2d([[1, 2], [3, 4]]).print();
 * ```
 * ```js
 * // Pass a flat array and specify a shape.
 * tf.tensor2d([1, 2, 3, 4], [2, 2]).print();
 * ```
 *
 * @param values The values of the tensor. Can be nested array of numbers,
 *     or a flat array, or a `TypedArray`.
 * @param shape The shape of the tensor. If not provided, it is inferred from
 *     `values`.
 * @param dtype The data type.
 *
 * @doc {heading: 'Tensors', subheading: 'Creation'}
 */
declare function tensor2d(values: TensorLike2D, shape?: [number, number], dtype?: DataType): Tensor2D;

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/tensor3d" />
/**
 * Creates rank-3 `tf.Tensor` with the provided values, shape and dtype.
 *
 * The same functionality can be achieved with `tf.tensor`, but in general
 * we recommend using `tf.tensor3d` as it makes the code more readable.
 *
 *  ```js
 * // Pass a nested array.
 * tf.tensor3d([[[1], [2]], [[3], [4]]]).print();
 * ```
 * ```js
 * // Pass a flat array and specify a shape.
 * tf.tensor3d([1, 2, 3, 4], [2, 2, 1]).print();
 * ```
 *
 * @param values The values of the tensor. Can be nested array of numbers,
 *     or a flat array, or a `TypedArray`.
 * @param shape The shape of the tensor. If not provided,  it is inferred from
 *     `values`.
 * @param dtype The data type.
 *
 * @doc {heading: 'Tensors', subheading: 'Creation'}
 */
declare function tensor3d(values: TensorLike3D, shape?: [number, number, number], dtype?: DataType): Tensor3D;

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/tensor4d" />
/**
 * Creates rank-4 `tf.Tensor` with the provided values, shape and dtype.
 *
 * The same functionality can be achieved with `tf.tensor`, but in general
 * we recommend using `tf.tensor4d` as it makes the code more readable.
 *
 *  ```js
 * // Pass a nested array.
 * tf.tensor4d([[[[1], [2]], [[3], [4]]]]).print();
 * ```
 * ```js
 * // Pass a flat array and specify a shape.
 * tf.tensor4d([1, 2, 3, 4], [1, 2, 2, 1]).print();
 * ```
 *
 * @param values The values of the tensor. Can be nested array of numbers,
 *     or a flat array, or a `TypedArray`.
 * @param shape The shape of the tensor. Optional. If not provided,
 *   it is inferred from `values`.
 * @param dtype The data type.
 *
 * @doc {heading: 'Tensors', subheading: 'Creation'}
 */
declare function tensor4d(values: TensorLike4D, shape?: [number, number, number, number], dtype?: DataType): Tensor4D;

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/tensor5d" />
/**
 * Creates rank-5 `tf.Tensor` with the provided values, shape and dtype.
 *
 * The same functionality can be achieved with `tf.tensor`, but in general
 * we recommend using `tf.tensor5d` as it makes the code more readable.
 *
 *  ```js
 * // Pass a nested array.
 * tf.tensor5d([[[[[1],[2]],[[3],[4]]],[[[5],[6]],[[7],[8]]]]]).print();
 * ```
 * ```js
 * // Pass a flat array and specify a shape.
 * tf.tensor5d([1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 2, 2, 1]).print();
 * ```
 *
 * @param values The values of the tensor. Can be nested array of numbers,
 *     or a flat array, or a `TypedArray`.
 * @param shape The shape of the tensor. Optional. If not provided,
 *   it is inferred from `values`.
 * @param dtype The data type.
 *
 * @doc {heading: 'Tensors', subheading: 'Creation'}
 */
declare function tensor5d(values: TensorLike5D, shape?: [number, number, number, number, number], dtype?: DataType): Tensor5D;

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/tensor6d" />
/**
 * Creates rank-6 `tf.Tensor` with the provided values, shape and dtype.
 *
 * The same functionality can be achieved with `tf.tensor`, but in general
 * we recommend using `tf.tensor6d` as it makes the code more readable.
 *
 *  ```js
 * // Pass a nested array.
 * tf.tensor6d([[[[[[1],[2]],[[3],[4]]],[[[5],[6]],[[7],[8]]]]]]).print();
 * ```
 * ```js
 * // Pass a flat array and specify a shape.
 * tf.tensor6d([1, 2, 3, 4, 5, 6, 7, 8], [1, 1, 2, 2, 2, 1]).print();
 * ```
 *
 * @param values The values of the tensor. Can be nested array of numbers,
 *     or a flat array, or a `TypedArray`.
 * @param shape The shape of the tensor. Optional. If not provided,
 *   it is inferred from `values`.
 * @param dtype The data type.
 *
 * @doc {heading: 'Tensors', subheading: 'Creation'}
 */
declare function tensor6d(values: TensorLike6D, shape?: [number, number, number, number, number, number], dtype?: DataType): Tensor6D;

/// <amd-module name="@tensorflow/tfjs-core/dist/tensor_format" />
declare function tensorToString(vals: TypedArray | string[], shape: number[], dtype: DataType, verbose: boolean): string;

/// <amd-module name="@tensorflow/tfjs-core/dist/tensor_info" />
/**
 * We wrap data id since we use weak map to avoid memory leaks.
 * Since we have our own memory management, we have a reference counter
 * mapping a tensor to its data, so there is always a pointer (even if that
 * data is otherwise garbage collectable).
 * See https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/
 * Global_Objects/WeakMap
 */
declare type DataId = object;
/** Holds metadata for a given tensor. */
interface TensorInfo {
    dataId: DataId;
    shape: number[];
    dtype: DataType;
}

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/tensor_ops_util" />
/** This is shared code across all tensor creation methods. */
declare function makeTensor(values: TensorLike | WebGLData | WebGPUData, shape: number[], inferredShape: number[], dtype?: DataType): Tensor;

/// <amd-module name="@tensorflow/tfjs-core/dist/tensor_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/tensor_types" />
/** @docalias {[name: string]: Tensor} */
declare type NamedTensorMap = {
    [name: string]: Tensor;
};
interface NamedTensor {
    name: string;
    tensor: Tensor;
}
declare type NamedVariableMap = {
    [name: string]: Variable;
};
declare type GradSaveFunc = (save: Tensor[]) => void;
/**
 * @docalias void|number|string|TypedArray|Tensor|Tensor[]|{[key:
 * string]:Tensor|number|string}
 */
declare type TensorContainer = void | Tensor | string | number | boolean | TensorContainerObject | TensorContainerArray | Float32Array | Int32Array | Uint8Array;
interface TensorContainerObject {
    [x: string]: TensorContainer;
}
interface TensorContainerArray extends Array<TensorContainer> {
}

/// <amd-module name="@tensorflow/tfjs-core/dist/tensor_util" />
declare function makeTypesMatch<T extends Tensor>(a: T, b: T): [T, T];
declare function assertTypesMatch(a: Tensor, b: Tensor): void;
declare function isTensorInList(tensor: Tensor, tensorList: Tensor[]): boolean;
/**
 * Extracts any `Tensor`s found within the provided object.
 *
 * @param container an object that may be a `Tensor` or may directly contain
 *   `Tensor`s, such as a `Tensor[]` or `{key: Tensor, ...}`. In general it
 *   is safe to pass any object here, except that `Promise`s are not
 *   supported.
 * @returns An array of `Tensors` found within the passed object. If the
 *   argument is simply a `Tensor', a list containing that `Tensor` is
 *   returned. If the object is not a `Tensor` or does not
 *   contain `Tensors`, an empty list is returned.
 */
declare function getTensorsInContainer(result: TensorContainer): Tensor[];

/// <amd-module name="@tensorflow/tfjs-core/dist/tensor_util_env" />
declare function inferShape(val: TensorLike | WebGLData | WebGPUData, dtype?: DataType): number[];
declare function convertToTensor<T extends Tensor>(x: T | TensorLike, argName: string, functionName: string, parseAsDtype?: DataType | 'numeric' | 'string_or_numeric'): T;
declare function convertToTensorArray<T extends Tensor>(arg: Array<T | TensorLike>, argName: string, functionName: string, parseAsDtype?: DataType | 'numeric' | 'string_or_numeric'): T[];

/// <amd-module name="@tensorflow/tfjs-core/dist/tensor_util_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/tests" />

/**
 * This file tests that we don't have any dataSyncs in the unconstrainted tests
 * so that we can run backends that have async init and async data reads against
 * our exported test files.
 */
/// <amd-module name="@tensorflow/tfjs-core/dist/test_async_backends" />

/// <amd-module name="@tensorflow/tfjs-data/dist/test_node" />

/// <amd-module name="@tensorflow/tfjs-core/dist/test_util" />
declare const TEST_EPSILON_FLOAT16 = 0.1;
declare function expectArraysClose(actual: TypedArray | number | RecursiveArray<number>, expected: TypedArray | number | RecursiveArray<number>, epsilon?: number): void;
declare function testEpsilon(): 0.001 | 0.1;
interface DoneFn {
    (): void;
    fail: (message?: Error | string) => void;
}
declare function expectPromiseToFail(fn: () => Promise<{}>, done: DoneFn): void;
declare function expectArraysEqual(actual: TensorLike, expected: TensorLike): void;
declare function expectNumbersClose(a: number, e: number, epsilon?: number): void;
declare function expectValuesInRange(actual: TypedArray | number[], low: number, high: number): void;
declare function expectArrayBuffersEqual(actual: ArrayBuffer, expected: ArrayBuffer): void;
/** Encodes strings into utf-8 bytes. */
declare function encodeStrings(a: RecursiveArray<{}>): RecursiveArray<Uint8Array>;
/** Creates an HTMLVideoElement with autoplay-friendly default settings. */
declare function createVideoElement(source: HTMLSourceElement): Promise<HTMLVideoElement>;
declare function play(video: HTMLVideoElement): Promise<void>;

/// <amd-module name="@tensorflow/tfjs-layers/dist/utils/test_utils" />
/**
 * Testing utilities.
 */
/**
 * Expect values are close between a Tensor or number array.
 * @param actual
 * @param expected
 */
declare function expectTensorsClose(actual: Tensor | number[], expected: Tensor | number[], epsilon?: number): void;
/**
 * Expect values in array are within a specified range, boundaries inclusive.
 * @param actual
 * @param expected
 */
declare function expectTensorsValuesInRange(actual: Tensor, low: number, high: number): void;
/**
 * Describe tests to be run on CPU and GPU.
 * @param testName
 * @param tests
 */
declare function describeMathCPUAndGPU(testName: string, tests: () => void): void;
/**
 * Describe tests to be run on CPU and GPU WebGL2.
 * @param testName
 * @param tests
 */
declare function describeMathCPUAndWebGL2(testName: string, tests: () => void): void;
/**
 * Describe tests to be run on CPU only.
 * @param testName
 * @param tests
 */
declare function describeMathCPU(testName: string, tests: () => void): void;
/**
 * Describe tests to be run on GPU only.
 * @param testName
 * @param tests
 */
declare function describeMathGPU(testName: string, tests: () => void): void;
/**
 * Describe tests to be run on WebGL2 GPU only.
 * @param testName
 * @param tests
 */
declare function describeMathWebGL2(testName: string, tests: () => void): void;
/**
 * Check that a function only generates the expected number of new Tensors.
 *
 * The test  function is called twice, once to prime any regular constants and
 * once to ensure that additional copies aren't created/tensors aren't leaked.
 *
 * @param testFunc A fully curried (zero arg) version of the function to test.
 * @param numNewTensors The expected number of new Tensors that should exist.
 */
declare function expectNoLeakedTensors(testFunc: () => any, numNewTensors: number): void;

/// <amd-module name="@tensorflow/tfjs-core/dist/test_util_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-data/dist/datasets/text_line_dataset" />
/**
 * Represents a potentially large collection of text lines.
 *
 * The results are not batched.
 */
declare class TextLineDataset extends Dataset<string> {
    protected readonly input: DataSource;
    /**
     * Create a `TextLineDataset`.
     *
     * @param input A `DataSource` providing a chunked, UTF8-encoded byte stream.
     */
    constructor(input: DataSource);
    iterator(): Promise<LazyIterator<string>>;
}

/// <amd-module name="@tensorflow/tfjs-layers/dist/backend/tfjs_backend" />
/**
 * deeplearn.js backend.
 */
declare function setBackend(requestedBackend: 'cpu' | 'webgl'): void;
declare function getBackend(): 'cpu' | 'webgl';
/**
 * Indicates whether the backend is operating symbolically.
 *
 * This function will be used to determine how to interpret user code. If
 * it returns true, calls to the backend construct a symbolic graph; if
 * it returns false, calls to the backend execute immediately.
 */
declare function isBackendSymbolic(): boolean;
/**
 * Get the number of elements in a Tensor.
 * @param x The Tensor.
 * @return Number of elements in `x`.
 */
declare function countParams(x: HasShape): number;
/**
 * Casts a tensor to a different dtype and returns it.
 * @param x Input tensor.
 * @param dtype String: 'float32'|'int32'|'bool'.
 * @returns Tensor of the specified `dtype`.
 */
declare function cast(x: Tensor, dtype: tfc.DataType): Tensor;
/**
 * Adds a 1-sized dimension at index "axis".
 * @param x Input tensor.
 * @param axis Position where to add the new axis.
 * @returns Result of the dimension expansion.
 */
declare function expandDims(x: Tensor, axis?: number): Tensor;
/**
 * Repeats a 2D tensor.
 *
 * If `x` has shape `[samples, dim]` and `n` is 2, for example, the output
 * will have shape `[samples, 2, dim]`.
 *
 * @param x Input tensor.
 * @param n Integer, number of times to repeat.
 * @returns The result of the repeat operation.
 * @throws ValueError: If input tensor is not 2D.
 */
declare function repeat(x: Tensor, n: number): Tensor;
/**
 * Flatten a Tensor into 1D.
 * @param x Input tensor.
 * @return The result of the flattening `x`.
 */
declare function flatten(x: Tensor): Tensor;
/**
 * Turn a nD tensor into a 2D tensor with same 0th dimension.
 * In other words, it flattens each data samples of a batch.
 *
 * @param x The tensor to flatten. The rank of this tensor is required to be 2
 *   or higher.
 * @return The result of the flattening.
 */
declare function batchFlatten(x: Tensor): Tensor;
/**
 * Do slicing along the first axis.
 * @param array input `tf.Tensor`.
 * @param start starting index, inclusive.
 * @param size size of the slice along the first axis.
 * @returns result of the slicing.
 * @throws ValueError: If `array` is of an unsupported subtype of `tf.Tensor`.
 */
declare function sliceAlongFirstAxis(array: Tensor, start: number, size: number): Tensor;
/**
 * Do slicing along the last axis.
 * @param array input `tf.Tensor`.
 * @param start starting index, inclusive.
 * @param size size of the slice along the last axis.
 * @returns result of the slicing.
 * @throws ValueError: If `array` is of an unsupported subtype of `tf.Tensor`.
 */
declare function sliceAlongLastAxis(array: Tensor, start: number, size: number): Tensor;
/**
 * Do slicing along the sepcified axis.
 * @param array input `tf.Tensor`.
 * @param start starting index, inclusive.
 * @param size of the slice along the chosen axis.
 * @param choose an axis.
 * @returns result of the slicing.
 * @throws ValueError: If `array` is of an unsupported subtype of `tf.Tensor`.
 */
declare function sliceAlongAxis(array: Tensor, start: number, size: number, axis: number): Tensor;
/**
 * Concatenates a list of tensors alongside the specified axis.
 * @param tensors `Array` of tensors to concatenate.
 * @param axis Concatenation axis.
 * @returns The result of the concatenation.
 */
declare function concatenate(tensors: Tensor[], axis?: number): Tensor;
/**
 * Concatenate two arrays along the first dimension.
 * @param a The 1st `tf.Tensor` to concatenate.
 * @param b The 2nd `tf.Tensor` to concatenate.
 * @returns Result of the concatenation.
 * @throws ValueError: If `a` is of an unsupported subtype of `tf.Tensor`.
 */
declare function concatAlongFirstAxis(a: Tensor, b: Tensor): Tensor;
/**
 * Creates a tensor by tiling `x` by `n`.
 * @param x A tensor.
 * @param n An Array of integers or a single integer. If an Array, the length
 *   must be the same as the number of dimensions in `x`. If a single integer,
 *   it will be treated as an Array of length 1.
 */
declare function tile(x: Tensor, n: number | number[]): Tensor;
/**
 * Get a tensor with normal distribution of values.
 *
 * @param shape Shape of the tensor.
 * @param mean mean value of the normal distribution.
 * @param stddev standard deviation of the normal distribution.
 * @param dtype
 * @param seed
 * @return The normal tensor.
 */
declare function randomNormal(shape: Shape, mean?: number, stddev?: number, dtype?: 'float32' | 'int32', seed?: number): Tensor;
/**
 * Multiply two tensors and returns the result as a tensor.
 *
 * For 2D tensors, this is equivalent to matrix multiplication (matMul).
 * For tensors of higher ranks, it follows the Theano behavior,
 * (e.g. `(2, 3) * (4, 3, 5) -> (2, 4, 5)`).  From the Theano documentation:
 *
 * For N dimensions it is a sum product over the last axis of x and the
 * second-to-last of y:
 *
 * @param a A tensor of at least rank 2.
 * @param b A tensor of at least rank 2.
 * @param activation (optional) A string identifying the activation
 *   function.
 * @return Result of the dot operation.
 */
declare function dot(a: Tensor, b: Tensor, activation?: tfc.fused.Activation, bias?: Tensor): Tensor;
/**
 * Compute the sign Tensor of an input Tensor.
 *
 * Elements of the input `tf.Tensor` that are === 0 are mapped to 0.
 * Elements of the input `tf.Tensor` that are > 0 are mapped to 1.
 * Elements of the input `tf.Tensor` that are < 0 are mapped to -1.
 *
 * @param x Input `tf.Tensor`.
 * @return The sign `tf.Tensor`.
 */
declare function sign(x: Tensor): Tensor;
/**
 * Computes the one-hot representation of an integer tensor.
 * @param indices nD integer tensor of shape
 *   `(batch_size, dim1, dim2, ... dim(n-1))`
 * @param numClasses Integer, number of classes to consider.
 * @returns (n + 1)D one hot representation of the input
 *   with shape `(batch_size, dim1, dim2, ... dim(n-1), num_classes)`
 */
declare function oneHot(indices: Tensor, numClasses: number): Tensor;
/**
 * Retrieves the elements of indices `indices` in the tensor `reference`.
 * @param reference A tensor.
 * @param indices An integer tensor of indices or an `Array` of integers.
 * @param axis Axis along which to perform the gather operation.
 * @returns The result of the gathering as a tensor.
 */
declare function gather(reference: Tensor, indices: number[] | Tensor1D, axis?: number): Tensor;
/**
 * Element-wise square.
 * @param x Input tensor.
 * @return element-wise x^2
 */
declare function square(x: Tensor): Tensor;
/**
 * Element-wise exponentiation.
 *
 * Porting Note: In PyKeras, `a` (the exponent) is a Python integer, which
 *   takes advatnage of the backend's (e.g., TensorFlow's) automatic
 * conversion to tensor. Here we allow `a` to be either a number or a tensor.
 *
 * @param x The base tensor.
 * @param a The exponent, tensor or number. If a number, it is rounded to the
 *   nearest integer and converted to a tensor.
 * @returns A tensor of the same shape as `x`.
 */
declare function pow(x: Tensor, a: Tensor | number): Tensor;
/**
 * Add a bias to a tensor.
 *
 * @param x The tensor to add the bias to.
 * @param bias The bias to add to `x`. Must be 1D or the same rank as `x`.
 * @return Result of the bias adding.
 * @throws ValueError: If the rank of `bias` is incorrect.
 */
declare function biasAdd(x: Tensor, bias: Tensor, dataFormat?: DataFormat): Tensor;
/**
 * Exponential linear unit (ELU).
 * @param x A tensor or variable to compute the activation function for.
 * @param alpha: A scalar, a scaling factor for the negative section.
 * @return Output of the ELU operation.
 */
declare function elu(x: Tensor, alpha?: number): Tensor;
/**
 * Softsign of a tensor.
 *
 * Defined as x / (abs(x) + 1), element-wise.
 *
 * @param x: Input.
 * @returns Output.
 */
declare function softsign(x: Tensor): Tensor;
/**
 * Sets entries in `x` to zero at random, while scaling the entire tensor.
 *
 * @param x input tensor.
 * @param level fraction of the entries in the tensor that will be set to 0.
 * @param noiseShape shape of randomly generated keep/drop flags, must be
 *   broadcastable to the shape of `x`. Optional.
 * @param seed random seed to ensure determinism. Optional.
 * @returns Result of the dropout operation.
 */
declare function dropout(x: Tensor, level: number, noiseShape?: number[], seed?: number): Tensor;
/**
 * Element-wise, segment-wise linear approximation of sigmoid.
 *
 * Returns `0.` if `x < -2.5`, `1.` if `x > 2.5`.
 * In `-2.5 <= x <= 2.5`, returns `0.2 * x + 0.5`.
 *
 * @param x Input tensor.
 * @returns Output tensor.
 */
declare function hardSigmoid(x: Tensor): Tensor;
/**
 * Invoke `x` in the training phase, and `alt` otherwise.
 *
 * Porting Note: We do not create placeholder tensors for the `training`
 * boolean flag here, because there is no such thing in the TF.js imperative
 * backend.
 *
 * @param x The function to invoke iff `training` is `true`.
 * @param alt The function to invoke iff `training` is `false`.
 * @param training Boolean flag for whether training phase is active.
 * @returns The return value of `x()` if `training` is `true`, or the return
 *   value of `alt()` if `training` is `false`.
 */
declare function inTrainPhase<T>(x: () => T, alt: () => T, training?: boolean): T;

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/image/threshold" />
/**
 * Performs image binarization with corresponding threshold
 * (depends on the method)value, which creates a binary image from a grayscale.
 * @param image 3d tensor of shape [imageHeight,imageWidth, depth],
 * where imageHeight and imageWidth must be positive.The image color
 * range should be [0, 255].
 * @param method Optional string from `'binary' | 'otsu'`
 * which specifies the method for thresholding. Defaults to 'binary'.
 * @param inverted Optional boolean whichspecifies
 * if colours should be inverted. Defaults to false.
 * @param threshValue Optional number which defines threshold value from 0 to 1.
 * Defaults to 0.5.
 * @return A 3d tensor of shape [imageHeight,imageWidth, depth], which
 * contains binarized image.
 */
declare function threshold_(image: Tensor3D | TensorLike, method?: string, inverted?: boolean, threshValue?: number): Tensor3D;
declare const threshold: typeof threshold_;
{ };
/// <amd-module name="@tensorflow/tfjs-core/dist/ops/image/threshold_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/tile" />
/**
 * Construct a tensor by repeating it the number of times given by reps.
 *
 * This operation creates a new tensor by replicating `input` `reps`
 * times. The output tensor's `i`th dimension has `input.shape[i] *
 * reps[i]` elements, and the values of `input` are replicated
 * `reps[i]` times along the `i`th dimension. For example, tiling
 * `[a, b, c, d]` by `[2]` produces `[a, b, c, d, a, b, c, d]`.
 *
 * ```js
 * const a = tf.tensor1d([1, 2]);
 *
 * a.tile([2]).print();    // or a.tile([2])
 * ```
 *
 * ```js
 * const a = tf.tensor2d([1, 2, 3, 4], [2, 2]);
 *
 * a.tile([1, 2]).print();  // or a.tile([1, 2])
 * ```
 * @param x The tensor to tile.
 * @param reps Determines the number of replications per dimension.
 *
 * @doc {heading: 'Tensors', subheading: 'Slicing and Joining'}
 */
declare function tile_<T extends Tensor>(x: T | TensorLike, reps: number[]): T;
declare const tile: typeof tile_;
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/gradients/Tile_grad" />
declare const tileGradConfig: GradConfig;

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/tile_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/public/chained_ops/topk" />
declare module '../../tensor' {
    interface Tensor<R extends Rank = Rank> {
        topk<T extends Tensor>(this: T, k?: number, sorted?: boolean): {
            values: T;
            indices: T;
        };
    }
}

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/topk_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-layers/dist/engine/topology" />
declare type Op = (x: LayerVariable) => LayerVariable;
/**
 * Constructor arguments for InputSpec.
 */
interface InputSpecArgs {
    /** Expected datatype of the input. */
    dtype?: DataType;
    /** Expected shape of the input (may include null for unchecked axes). */
    shape?: Shape;
    /** Expected rank of the input. */
    ndim?: number;
    /** Maximum rank of the input. */
    maxNDim?: number;
    /** Minimum rank of the input. */
    minNDim?: number;
    /** Dictionary mapping integer axes to a specific dimension value. */
    axes?: {
        [axis: number]: number;
    };
}
/**
 * Specifies the ndim, dtype and shape of every input to a layer.
 *
 * Every layer should expose (if appropriate) an `inputSpec` attribute:
 * a list of instances of InputSpec (one per input tensor).
 *
 * A null entry in a shape is compatible with any dimension,
 * a null shape is compatible with any shape.
 */
declare class InputSpec {
    /** Expected datatype of the input. */
    dtype?: DataType;
    /** Expected shape of the input (may include null for unchecked axes). */
    shape?: Shape;
    /** Expected rank of the input. */
    ndim?: number;
    /** Maximum rank of the input. */
    maxNDim?: number;
    /** Minimum rank of the input. */
    minNDim?: number;
    /** Dictionary mapping integer axes to a specific dimension value. */
    axes?: {
        [axis: number]: number;
    };
    constructor(args: InputSpecArgs);
}
/**
 * `tf.SymbolicTensor` is a placeholder for a Tensor without any concrete value.
 *
 * They are most often encountered when building a graph of `Layer`s for a
 * `tf.LayersModel` and the input data's shape, but not values are known.
 *
 * @doc {heading: 'Models', 'subheading': 'Classes'}
 */
declare class SymbolicTensor {
    readonly dtype: DataType;
    readonly shape: Shape;
    sourceLayer: Layer;
    readonly inputs: SymbolicTensor[];
    readonly callArgs: Kwargs;
    readonly outputTensorIndex?: number;
    readonly id: number;
    readonly name: string;
    readonly originalName?: string;
    /**
     * Rank/dimensionality of the tensor.
     */
    readonly rank: number;
    /**
     * Replacement for _keras_history.
     */
    nodeIndex: number;
    /**
     * Replacement for _keras_history.
     */
    tensorIndex: number;
    /**
     *
     * @param dtype
     * @param shape
     * @param sourceLayer The Layer that produced this symbolic tensor.
     * @param inputs The inputs passed to sourceLayer's __call__() method.
     * @param nodeIndex
     * @param tensorIndex
     * @param callArgs The keyword arguments passed to the __call__() method.
     * @param name
     * @param outputTensorIndex The index of this tensor in the list of outputs
     *   returned by apply().
     */
    constructor(dtype: DataType, shape: Shape, sourceLayer: Layer, inputs: SymbolicTensor[], callArgs: Kwargs, name?: string, outputTensorIndex?: number);
}
/**
 * Constructor arguments for Node.
 */
interface NodeArgs {
    /**
     * The layer that takes `inputTensors` and turns them into `outputTensors`.
     * (the node gets created when the `call` method of the layer is called).
     */
    outboundLayer: Layer;
    /**
     * A list of layers, the same length as `inputTensors`, the layers from where
     * `inputTensors` originate.
     */
    inboundLayers: Layer[];
    /**
     * A list of integers, the same length as `inboundLayers`. `nodeIndices[i]` is
     * the origin node of `inputTensors[i]` (necessary since each inbound layer
     * might have several nodes, e.g. if the layer is being shared with a
     * different data stream).
     */
    nodeIndices: number[];
    /**
     * A list of integers, the same length as `inboundLayers`. `tensorIndices[i]`
     * is the index of `inputTensors[i]` within the output of the inbound layer
     * (necessary since each inbound layer might have multiple tensor outputs,
     * with each one being independently manipulable).
     */
    tensorIndices: number[];
    /** List of input tensors. */
    inputTensors: SymbolicTensor[];
    /** List of output tensors. */
    outputTensors: SymbolicTensor[];
    /** List of input masks (a mask can be a tensor, or null). */
    inputMasks: Tensor[];
    /** List of output masks (a mask can be a tensor, or null). */
    outputMasks: Tensor[];
    /** List of input shape tuples. */
    inputShapes: Shape | Shape[];
    /** List of output shape tuples. */
    outputShapes: Shape | Shape[];
}
/**
 * The type of the return value of Layer.dispose() and Container.dispose().
 */
interface DisposeResult {
    /**
     * Reference count after the dispose call.
     */
    refCountAfterDispose: number;
    /**
     * Number of variables dispose in this dispose call.
     */
    numDisposedVariables: number;
}
/**
 * A `Node` describes the connectivity between two layers.
 *
 * Each time a layer is connected to some new input,
 * a node is added to `layer.inboundNodes`.
 *
 * Each time the output of a layer is used by another layer,
 * a node is added to `layer.outboundNodes`.
 *
 * `nodeIndices` and `tensorIndices` are basically fine-grained coordinates
 * describing the origin of the `inputTensors`, verifying the following:
 *
 * `inputTensors[i] ==
 * inboundLayers[i].inboundNodes[nodeIndices[i]].outputTensors[
 *   tensorIndices[i]]`
 *
 * A node from layer A to layer B is added to:
 *     A.outboundNodes
 *     B.inboundNodes
 */
declare class Node {
    callArgs?: Kwargs;
    /**
     * The layer that takes `inputTensors` and turns them into `outputTensors`
     * (the node gets created when the `call` method of the layer is called).
     */
    outboundLayer: Layer;
    /**
     * A list of layers, the same length as `inputTensors`, the layers from where
     * `inputTensors` originate.
     */
    inboundLayers: Layer[];
    /**
     * A list of integers, the same length as `inboundLayers`. `nodeIndices[i]` is
     * the origin node of `inputTensors[i]` (necessary since each inbound layer
     * might have several nodes, e.g. if the layer is being shared with a
     * different data stream).
     */
    nodeIndices: number[];
    /**
     * A list of integers, the same length as `inboundLayers`. `tensorIndices[i]`
     * is the index of `inputTensors[i]` within the output of the inbound layer
     * (necessary since each inbound layer might have multiple tensor outputs,
     * with each one being independently manipulable).
     */
    tensorIndices: number[];
    /** List of input tensors. */
    inputTensors: SymbolicTensor[];
    /** List of output tensors. */
    outputTensors: SymbolicTensor[];
    /** List of input masks (a mask can be a tensor, or null). */
    inputMasks: Tensor[];
    /** List of output masks (a mask can be a tensor, or null). */
    outputMasks: Tensor[];
    /** List of input shape tuples. */
    inputShapes: Shape | Shape[];
    /** List of output shape tuples. */
    outputShapes: Shape | Shape[];
    readonly id: number;
    constructor(args: NodeArgs, callArgs?: Kwargs);
    getConfig(): serialization.ConfigDict;
}
/** Constructor arguments for Layer. */
declare interface LayerArgs {
    /**
     * If defined, will be used to create an input layer to insert before this
     * layer. If both `inputShape` and `batchInputShape` are defined,
     * `batchInputShape` will be used. This argument is only applicable to input
     * layers (the first layer of a model).
     */
    inputShape?: Shape;
    /**
     * If defined, will be used to create an input layer to insert before this
     * layer. If both `inputShape` and `batchInputShape` are defined,
     * `batchInputShape` will be used. This argument is only applicable to input
     * layers (the first layer of a model).
     */
    batchInputShape?: Shape;
    /**
     * If `inputShape` is specified and `batchInputShape` is *not* specified,
     * `batchSize` is used to construct the `batchInputShape`: `[batchSize,
     * ...inputShape]`
     */
    batchSize?: number;
    /**
     * The data-type for this layer. Defaults to 'float32'.
     * This argument is only applicable to input layers (the first layer of a
     * model).
     */
    dtype?: DataType;
    /** Name for this layer. */
    name?: string;
    /**
     * Whether the weights of this layer are updatable by `fit`.
     * Defaults to true.
     */
    trainable?: boolean;
    /**
     * Initial weight values of the layer.
     */
    weights?: Tensor[];
    /** Legacy support. Do not use for new code. */
    inputDType?: DataType;
}
declare type CallHook = (inputs: Tensor | Tensor[], kwargs: Kwargs) => void;
/**
 * A layer is a grouping of operations and weights that can be composed to
 * create a `tf.LayersModel`.
 *
 * Layers are constructed by using the functions under the
 * [tf.layers](#Layers-Basic) namespace.
 *
 * @doc {heading: 'Layers', subheading: 'Classes', namespace: 'layers'}
 */
declare abstract class Layer extends serialization.Serializable {
    /** Name for this layer. Must be unique within a model. */
    name: string;
    /**
     * List of InputSpec class instances.
     *
     * Each entry describes one required input:
     * - ndim
     * - dtype
     * A layer with `n` input tensors must have an `inputSpec` of length `n`.
     */
    inputSpec: InputSpec[];
    supportsMasking: boolean;
    /** Whether the layer weights will be updated during training. */
    protected trainable_: boolean;
    batchInputShape: Shape;
    dtype: DataType;
    initialWeights: Tensor[];
    inboundNodes: Node[];
    outboundNodes: Node[];
    activityRegularizer: Regularizer;
    protected _trainableWeights: LayerVariable[];
    private _nonTrainableWeights;
    private _losses;
    private _updates;
    private _built;
    private _callHook;
    private _addedWeightNames;
    readonly id: number;
    protected _stateful: boolean;
    protected _refCount: number | null;
    private fastWeightInitDuringBuild;
    constructor(args?: LayerArgs);
    /**
     * Converts a layer and its index to a unique (immutable type) name.
     * This function is used internally with `this.containerNodes`.
     * @param layer The layer.
     * @param nodeIndex The layer's position (e.g. via enumerate) in a list of
     *   nodes.
     *
     * @returns The unique name.
     */
    protected static nodeKey(layer: Layer, nodeIndex: number): string;
    /**
     * Returns this.inboundNode at index nodeIndex.
     *
     * Porting note: This is a replacement for _get_node_attribute_at_index()
     * @param nodeIndex
     * @param attrName The name of the attribute related to request for this node.
     */
    private getNodeAtIndex;
    /**
     * Retrieves the input tensor(s) of a layer at a given node.
     *
     * @param nodeIndex Integer, index of the node from which to retrieve the
     *   attribute. E.g. `nodeIndex=0` will correspond to the first time the layer
     *   was called.
     *
     * @return A tensor (or list of tensors if the layer has multiple inputs).
     */
    getInputAt(nodeIndex: number): SymbolicTensor | SymbolicTensor[];
    /**
     * Retrieves the output tensor(s) of a layer at a given node.
     *
     * @param nodeIndex Integer, index of the node from which to retrieve the
     *   attribute. E.g. `nodeIndex=0` will correspond to the first time the layer
     *   was called.
     *
     * @return A tensor (or list of tensors if the layer has multiple outputs).
     */
    getOutputAt(nodeIndex: number): SymbolicTensor | SymbolicTensor[];
    /**
     * Retrieves the input tensor(s) of a layer.
     *
     * Only applicable if the layer has exactly one inbound node,
     * i.e. if it is connected to one incoming layer.
     *
     * @return Input tensor or list of input tensors.
     *
     * @exception AttributeError if the layer is connected to more than one
     *   incoming layers.
     */
    get input(): SymbolicTensor | SymbolicTensor[];
    /**
     * Retrieves the output tensor(s) of a layer.
     *
     * Only applicable if the layer has exactly one inbound node,
     * i.e. if it is connected to one incoming layer.
     *
     * @return Output tensor or list of output tensors.
     *
     * @exception AttributeError if the layer is connected to more than one
     *   incoming layers.
     */
    get output(): SymbolicTensor | SymbolicTensor[];
    get losses(): RegularizerFn[];
    /**
     * Retrieves the Layer's current loss values.
     *
     * Used for regularizers during training.
     */
    calculateLosses(): Scalar[];
    get updates(): Tensor[];
    get built(): boolean;
    set built(built: boolean);
    get trainable(): boolean;
    set trainable(trainable: boolean);
    get trainableWeights(): LayerVariable[];
    set trainableWeights(weights: LayerVariable[]);
    get nonTrainableWeights(): LayerVariable[];
    set nonTrainableWeights(weights: LayerVariable[]);
    /**
     * The concatenation of the lists trainableWeights and nonTrainableWeights
     * (in this order).
     */
    get weights(): LayerVariable[];
    get stateful(): boolean;
    /**
     * Reset the states of the layer.
     *
     * This method of the base Layer class is essentially a no-op.
     * Subclasses that are stateful (e.g., stateful RNNs) should override this
     * method.
     */
    resetStates(): void;
    /**
     * Checks compatibility between the layer and provided inputs.
     *
     * This checks that the tensor(s) `input`
     * verify the input assumptions of the layer
     * (if any). If not, exceptions are raised.
     *
     * @param inputs Input tensor or list of input tensors.
     *
     * @exception ValueError in case of mismatch between
     *   the provided inputs and the expectations of the layer.
     */
    protected assertInputCompatibility(inputs: Tensor | Tensor[] | SymbolicTensor | SymbolicTensor[]): void;
    /**
     * This is where the layer's logic lives.
     *
     * @param inputs Input tensor, or list/tuple of input tensors.
     * @param kwargs Additional keyword arguments.
     *
     * @return A tensor or list/tuple of tensors.
     */
    call(inputs: Tensor | Tensor[], kwargs: Kwargs): Tensor | Tensor[];
    protected invokeCallHook(inputs: Tensor | Tensor[], kwargs: Kwargs): void;
    /**
     * Set call hook.
     * This is currently used for testing only.
     * @param callHook
     */
    setCallHook(callHook: CallHook): void;
    /**
     * Clear call hook.
     * This is currently used for testing only.
     */
    clearCallHook(): void;
    /**
     * Builds or executes a `Layer`'s logic.
     *
     * When called with `tf.Tensor`(s), execute the `Layer`'s computation and
     * return Tensor(s). For example:
     *
     * ```js
     * const denseLayer = tf.layers.dense({
     *   units: 1,
     *   kernelInitializer: 'zeros',
     *   useBias: false
     * });
     *
     * // Invoke the layer's apply() method with a `tf.Tensor` (with concrete
     * // numeric values).
     * const input = tf.ones([2, 2]);
     * const output = denseLayer.apply(input);
     *
     * // The output's value is expected to be [[0], [0]], due to the fact that
     * // the dense layer has a kernel initialized to all-zeros and does not have
     * // a bias.
     * output.print();
     * ```
     *
     * When called with `tf.SymbolicTensor`(s), this will prepare the layer for
     * future execution.  This entails internal book-keeping on shapes of
     * expected Tensors, wiring layers together, and initializing weights.
     *
     * Calling `apply` with `tf.SymbolicTensor`s are typically used during the
     * building of non-`tf.Sequential` models. For example:
     *
     * ```js
     * const flattenLayer = tf.layers.flatten();
     * const denseLayer = tf.layers.dense({units: 1});
     *
     * // Use tf.layers.input() to obtain a SymbolicTensor as input to apply().
     * const input = tf.input({shape: [2, 2]});
     * const output1 = flattenLayer.apply(input);
     *
     * // output1.shape is [null, 4]. The first dimension is the undetermined
     * // batch size. The second dimension comes from flattening the [2, 2]
     * // shape.
     * console.log(JSON.stringify(output1.shape));
     *
     * // The output SymbolicTensor of the flatten layer can be used to call
     * // the apply() of the dense layer:
     * const output2 = denseLayer.apply(output1);
     *
     * // output2.shape is [null, 1]. The first dimension is the undetermined
     * // batch size. The second dimension matches the number of units of the
     * // dense layer.
     * console.log(JSON.stringify(output2.shape));
     *
     * // The input and output can be used to construct a model that consists
     * // of the flatten and dense layers.
     * const model = tf.model({inputs: input, outputs: output2});
     * ```
     *
     * @param inputs a `tf.Tensor` or `tf.SymbolicTensor` or an Array of them.
     * @param kwargs Additional keyword arguments to be passed to `call()`.
     *
     * @return Output of the layer's `call` method.
     *
     * @exception ValueError error in case the layer is missing shape information
     *   for its `build` call.
     *
     * @doc {heading: 'Models', 'subheading': 'Classes'}
     */
    apply(inputs: Tensor | Tensor[] | SymbolicTensor | SymbolicTensor[], kwargs?: Kwargs): Tensor | Tensor[] | SymbolicTensor | SymbolicTensor[];
    /**
     * Check compatibility between input shape and this layer's batchInputShape.
     *
     * Print warning if any incompatibility is found.
     *
     * @param inputShape Input shape to be checked.
     */
    protected warnOnIncompatibleInputShape(inputShape: Shape): void;
    /**
     * Retrieves the output shape(s) of a layer.
     *
     * Only applicable if the layer has only one inbound node, or if all inbound
     * nodes have the same output shape.
     *
     * @returns Output shape or shapes.
     * @throws AttributeError: if the layer is connected to more than one incoming
     *   nodes.
     *
     * @doc {heading: 'Models', 'subheading': 'Classes'}
     */
    get outputShape(): Shape | Shape[];
    /**
     * Counts the total number of numbers (e.g., float32, int32) in the
     * weights.
     *
     * @returns An integer count.
     * @throws RuntimeError: If the layer is not built yet (in which case its
     *   weights are not defined yet.)
     *
     * @doc {heading: 'Models', 'subheading': 'Classes'}
     */
    countParams(): number;
    /**
     * Creates the layer weights.
     *
     * Must be implemented on all layers that have weights.
     *
     * Called when apply() is called to construct the weights.
     *
     * @param inputShape A `Shape` or array of `Shape` (unused).
     *
     * @doc {heading: 'Models', 'subheading': 'Classes'}
     */
    build(inputShape: Shape | Shape[]): void;
    /**
     * Returns the current values of the weights of the layer.
     *
     * @param trainableOnly Whether to get the values of only trainable weights.
     * @returns Weight values as an `Array` of `tf.Tensor`s.
     *
     * @doc {heading: 'Models', 'subheading': 'Classes'}
     */
    getWeights(trainableOnly?: boolean): Tensor[];
    /**
     * Sets the weights of the layer, from Tensors.
     *
     * @param weights a list of Tensors. The number of arrays and their shape
     *   must match number of the dimensions of the weights of the layer (i.e.
     *   it should match the output of `getWeights`).
     *
     * @exception ValueError If the provided weights list does not match the
     *   layer's specifications.
     *
     * @doc {heading: 'Models', 'subheading': 'Classes'}
     */
    setWeights(weights: Tensor[]): void;
    /**
     * Adds a weight variable to the layer.
     *
     * @param name Name of the new weight variable.
     * @param shape The shape of the weight.
     * @param dtype The dtype of the weight.
     * @param initializer An initializer instance.
     * @param regularizer A regularizer instance.
     * @param trainable Whether the weight should be trained via backprop or not
     *   (assuming that the layer itself is also trainable).
     * @param constraint An optional trainable.
     * @return The created weight variable.
     *
     * @doc {heading: 'Models', 'subheading': 'Classes'}
     */
    protected addWeight(name: string, shape: Shape, dtype?: DataType, initializer?: Initializer, regularizer?: Regularizer, trainable?: boolean, constraint?: Constraint, getInitializerFunc?: Function): LayerVariable;
    /**
     * Set the fast-weight-initialization flag.
     *
     * In cases where the initialized weight values will be immediately
     * overwritten by loaded weight values during model loading, setting
     * the flag to `true` saves unnecessary calls to potentially expensive
     * initializers and speeds up the loading process.
     *
     * @param value Target value of the flag.
     */
    setFastWeightInitDuringBuild(value: boolean): void;
    /**
     * Add losses to the layer.
     *
     * The loss may potentially be conditional on some inputs tensors,
     * for instance activity losses are conditional on the layer's inputs.
     *
     * @doc {heading: 'Models', 'subheading': 'Classes'}
     */
    addLoss(losses: RegularizerFn | RegularizerFn[]): void;
    /**
     * Computes the output shape of the layer.
     *
     * Assumes that the layer will be built to match that input shape provided.
     *
     * @param inputShape A shape (tuple of integers) or a list of shape tuples
     *   (one per output tensor of the layer). Shape tuples can include null for
     *   free dimensions, instead of an integer.
     *
     * @doc {heading: 'Models', 'subheading': 'Classes'}
     */
    computeOutputShape(inputShape: Shape | Shape[]): Shape | Shape[];
    /**
     * Computes an output mask tensor.
     *
     * @param inputs Tensor or list of tensors.
     * @param mask Tensor or list of tensors.
     *
     * @return null or a tensor (or list of tensors, one per output tensor of the
     * layer).
     */
    computeMask(inputs: Tensor | Tensor[], mask?: Tensor | Tensor[]): Tensor | Tensor[];
    /**
     * Internal method to create an inbound node for the layer.
     *
     * @param inputTensors List of input tensors.
     * @param outputTensors List of output tensors.
     * @param inputMasks List of input masks (a mask can be a tensor, or null).
     * @param outputMasks List of output masks (a mask can be a tensor, or null).
     * @param inputShapes List of input shape tuples.
     * @param outputShapes List of output shape tuples.
     * @param kwargs Dictionary of keyword arguments that were passed to the
     *   `call` method of the layer at the call that created the node.
     */
    private addInboundNode;
    /**
     * Returns the config of the layer.
     *
     * A layer config is a TS dictionary (serializable)
     * containing the configuration of a layer.
     * The same layer can be reinstantiated later
     * (without its trained weights) from this configuration.
     *
     * The config of a layer does not include connectivity
     * information, nor the layer class name.  These are handled
     * by 'Container' (one layer of abstraction above).
     *
     * Porting Note: The TS dictionary follows TS naming standards for
     * keys, and uses tfjs-layers type-safe Enums.  Serialization methods
     * should use a helper function to convert to the pythonic storage
     * standard. (see serialization_utils.convertTsToPythonic)
     *
     * @returns TS dictionary of configuration.
     *
     * @doc {heading: 'Models', 'subheading': 'Classes'}
     */
    getConfig(): serialization.ConfigDict;
    /**
     * Dispose the weight variables that this Layer instance holds.
     *
     * @returns {number} Number of disposed variables.
     */
    protected disposeWeights(): number;
    protected assertNotDisposed(): void;
    /**
     * Attempt to dispose layer's weights.
     *
     * This method decreases the reference count of the Layer object by 1.
     *
     * A Layer is reference-counted. Its reference count is incremented by 1
     * the first item its `apply()` method is called and when it becomes a part
     * of a new `Node` (through calling the `apply()` method on a
     * `tf.SymbolicTensor`).
     *
     * If the reference count of a Layer becomes 0, all the weights will be
     * disposed and the underlying memory (e.g., the textures allocated in WebGL)
     * will be freed.
     *
     * Note: If the reference count is greater than 0 after the decrement, the
     * weights of the Layer will *not* be disposed.
     *
     * After a Layer is disposed, it cannot be used in calls such as `apply()`,
     * `getWeights()` or `setWeights()` anymore.
     *
     * @returns A DisposeResult Object with the following fields:
     *   - refCountAfterDispose: The reference count of the Container after this
     *     `dispose()` call.
     *   - numDisposedVariables: Number of `tf.Variable`s (i.e., weights) disposed
     *     during this `dispose()` call.
     * @throws {Error} If the layer is not built yet, or if the layer has already
     *   been disposed.
     *
     * @doc {heading: 'Models', 'subheading': 'Classes'}
     */
    dispose(): DisposeResult;
}
/**
 * Returns the list of input tensors necessary to compute `tensor`.
 *
 * Output will always be a list of tensors (potentially with 1 element).
 *
 * @param tensor The tensor to start from.
 * @param layer Origin layer of the tensor.
 * @param nodeIndex Origin node index of the tensor.
 *
 * @return Array of input tensors.
 */
declare function getSourceInputs(tensor: SymbolicTensor, layer?: Layer, nodeIndex?: number): SymbolicTensor[];

/// <amd-module name="@tensorflow/tfjs-layers/dist/keras_format/topology_config" />
/** Constructor arguments for Layer. */
interface LayerConfig extends PyJsonDict {
    input_shape?: Shape;
    batch_input_shape?: Shape;
    batch_size?: number;
    dtype?: DataType;
    name?: string;
    trainable?: boolean;
    input_dtype?: DataType;
}
/**
 * Converts a subtype of `LayerConfig` to a variant with restricted keys.
 *
 * This is a bit tricky because `keyof` obtains only local fields, not inherited
 * fields.  Thus, this type combines the keys from the `LayerConfig` supertype
 * with those of the specific subtype.
 *
 * See ./types.ts for an explanation of the PyJson type.
 */
declare type JsonLayer<C extends LayerConfig> = C & LayerConfig & PyJson<Extract<keyof C, string> | Extract<keyof LayerConfig, string>>;
/**
 * A Keras JSON entry representing a layer.
 *
 * The Keras JSON convention is to provide the `class_name` (i.e., the layer
 * type) at the top level, and then to place the layer-specific configuration in
 * a `config` subtree.  These layer-specific configurations are provided by
 * subtypes of `LayerConfig`.  Thus, this `*Serialization` has a type parameter
 * giving the specific type of the wrapped `LayerConfig`.
 */
interface BaseLayerSerialization<N extends string, C extends LayerConfig> extends BaseSerialization<N, JsonLayer<C>> {
    name: string;
    inbound_nodes?: NodeConfig[];
}

/// <amd-module name="@tensorflow/tfjs-core/dist/public/chained_ops/to_bool" />
declare module '../../tensor' {
    interface Tensor<R extends Rank = Rank> {
        toBool<T extends Tensor>(this: T): T;
    }
}

/// <amd-module name="@tensorflow/tfjs-core/dist/public/chained_ops/to_float" />
declare module '../../tensor' {
    interface Tensor<R extends Rank = Rank> {
        toFloat<T extends Tensor>(this: T): T;
    }
}

/// <amd-module name="@tensorflow/tfjs-core/dist/public/chained_ops/to_int" />
declare module '../../tensor' {
    interface Tensor<R extends Rank = Rank> {
        toInt<T extends Tensor>(this: T): T;
    }
}

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/to_pixels_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/train" />
declare const train: typeof OptimizerConstructors;

/// <amd-module name="@tensorflow/tfjs-layers/dist/engine/training" />
/**
 * Helper function for polymorphic input data: 1. singleton Tensor.
 */
declare function isDataTensor(x: Tensor | Tensor[] | {
    [inputName: string]: Tensor;
} | {
    [inputName: string]: Tensor[];
}): boolean;
/**
 * Helper function for polymorphic input data: 2. Array of Tensor.
 */
declare function isDataArray(x: Tensor | Tensor[] | {
    [inputName: string]: Tensor;
}): boolean;
/**
 * Helper function for polymorphic input data: 3. "dict" of Tensor.
 */
declare function isDataDict(x: Tensor | Tensor[] | {
    [inputName: string]: Tensor;
}): boolean;
/**
 * Normalizes inputs and targets provided by users.
 * @param data User-provided input data (polymorphic).
 * @param names An Array of expected Tensor names.
 * @param shapes Optional Array of expected Tensor shapes.
 * @param checkBatchAxis Whether to check that the batch axis of the arrays
 *   match  the expected value found in `shapes`.
 * @param exceptionPrefix String prefix used for exception formatting.
 * @returns List of standardized input Tensors (one Tensor per model input).
 * @throws ValueError: in case of improperly formatted user data.
 */
declare function standardizeInputData(data: Tensor | Tensor[] | {
    [inputName: string]: Tensor;
}, names: string[], shapes?: Shape[], checkBatchAxis?: boolean, exceptionPrefix?: string): Tensor[];
/**
 * User input validation for Tensors.
 * @param inputs `Array` of `tf.Tensor`s for inputs.
 * @param targets `Array` of `tf.Tensor`s for targets.
 * @param weights Optional `Array` of `tf.Tensor`s for sample weights.
 * @throws ValueError: in case of incorrectly formatted data.
 */
declare function checkArrayLengths(inputs: Tensor[], targets: Tensor[], weights?: Tensor[]): void;
/**
 * Maps metric functions to model outputs.
 * @param metrics An shortcut strings name, metric function, `Array` or dict
 *   (`Object`) of metric functions.
 * @param outputNames An `Array` of the names of model outputs.
 * @returns An `Array` (one entry per model output) of `Array` of metric
 *   functions. For instance, if the model has 2 outputs, and for the first
 *   output we want to compute `binaryAccuracy` and `binaryCrossentropy`,
 *   and just `binaryAccuracy` for the second output, the `Array` would look
 *   like:
 *     `[[binaryAccuracy, binaryCrossentropy],  [binaryAccuracy]]`
 * @throws TypeError: incompatible metrics format.
 */
declare function collectMetrics(metrics: string | LossOrMetricFn | Array<string | LossOrMetricFn> | {
    [outputName: string]: string | LossOrMetricFn;
}, outputNames: string[]): Array<Array<string | LossOrMetricFn>>;
interface ModelEvaluateArgs {
    /**
     * Batch size (Integer). If unspecified, it will default to 32.
     */
    batchSize?: number;
    /**
     * Verbosity mode.
     */
    verbose?: ModelLoggingVerbosity;
    /**
     * Tensor of weights to weight the contribution of different samples to the
     * loss and metrics.
     */
    sampleWeight?: Tensor;
    /**
     * integer: total number of steps (batches of samples)
     * before declaring the evaluation round finished. Ignored with the default
     * value of `undefined`.
     */
    steps?: number;
}
/**
 * Configuration for calls to `LayersModel.compile()`.
 */
interface ModelCompileArgs {
    /**
     * An instance of `tf.train.Optimizer` or a string name for an Optimizer.
     */
    optimizer: string | Optimizer;
    /**
     * Object function(s) or name(s) of object function(s).
     * If the model has multiple outputs, you can use a different loss
     * on each output by passing a dictionary or an Array of losses.
     * The loss value that will be minimized by the model will then be the sum
     * of all individual losses.
     */
    loss: string | string[] | {
        [outputName: string]: string;
    } | LossOrMetricFn | LossOrMetricFn[] | {
        [outputName: string]: LossOrMetricFn;
    };
    /**
     * List of metrics to be evaluated by the model during training and testing.
     * Typically you will use `metrics=['accuracy']`.
     * To specify different metrics for different outputs of a multi-output
     * model, you could also pass a dictionary.
     */
    metrics?: string | LossOrMetricFn | Array<string | LossOrMetricFn> | {
        [outputName: string]: string | LossOrMetricFn;
    };
}
/**
 * A `tf.LayersModel` is a directed, acyclic graph of `tf.Layer`s plus methods
 * for training, evaluation, prediction and saving.
 *
 * `tf.LayersModel` is the basic unit of training, inference and evaluation in
 * TensorFlow.js. To create a `tf.LayersModel`, use `tf.LayersModel`.
 *
 * See also:
 *   `tf.Sequential`, `tf.loadLayersModel`.
 *
 * @doc {heading: 'Models', subheading: 'Classes'}
 */
declare class LayersModel extends Container implements tfc.InferenceModel {
    /** @nocollapse */
    static className: string;
    protected optimizer_: Optimizer;
    protected isOptimizerOwned: boolean;
    loss: string | string[] | {
        [outputName: string]: string;
    } | LossOrMetricFn | LossOrMetricFn[] | {
        [outputName: string]: LossOrMetricFn;
    };
    lossFunctions: LossOrMetricFn[];
    private feedOutputShapes;
    private feedLossFns;
    private collectedTrainableWeights;
    private testFunction;
    history: History;
    protected stopTraining_: boolean;
    protected isTraining: boolean;
    metrics: string | LossOrMetricFn | Array<string | LossOrMetricFn> | {
        [outputName: string]: string | LossOrMetricFn;
    };
    metricsNames: string[];
    metricsTensors: Array<[LossOrMetricFn, number]>;
    private userDefinedMetadata;
    constructor(args: ContainerArgs);
    /**
     * Print a text summary of the model's layers.
     *
     * The summary includes
     * - Name and type of all layers that comprise the model.
     * - Output shape(s) of the layers
     * - Number of weight parameters of each layer
     * - If the model has non-sequential-like topology, the inputs each layer
     *   receives
     * - The total number of trainable and non-trainable parameters of the model.
     *
     * ```js
     * const input1 = tf.input({shape: [10]});
     * const input2 = tf.input({shape: [20]});
     * const dense1 = tf.layers.dense({units: 4}).apply(input1);
     * const dense2 = tf.layers.dense({units: 8}).apply(input2);
     * const concat = tf.layers.concatenate().apply([dense1, dense2]);
     * const output =
     *     tf.layers.dense({units: 3, activation: 'softmax'}).apply(concat);
     *
     * const model = tf.model({inputs: [input1, input2], outputs: output});
     * model.summary();
     * ```
     *
     * @param lineLength Custom line length, in number of characters.
     * @param positions Custom widths of each of the columns, as either
     *   fractions of `lineLength` (e.g., `[0.5, 0.75, 1]`) or absolute number
     *   of characters (e.g., `[30, 50, 65]`). Each number corresponds to
     *   right-most (i.e., ending) position of a column.
     * @param printFn Custom print function. Can be used to replace the default
     *   `console.log`. For example, you can use `x => {}` to mute the printed
     *   messages in the console.
     *
     * @doc {heading: 'Models', subheading: 'Classes'}
     */
    summary(lineLength?: number, positions?: number[], printFn?: (message?: any, ...optionalParams: any[]) => void): void;
    /**
     * Configures and prepares the model for training and evaluation.  Compiling
     * outfits the model with an optimizer, loss, and/or metrics.  Calling `fit`
     * or `evaluate` on an un-compiled model will throw an error.
     *
     * @param args a `ModelCompileArgs` specifying the loss, optimizer, and
     * metrics to be used for fitting and evaluating this model.
     *
     * @doc {heading: 'Models', subheading: 'Classes'}
     */
    compile(args: ModelCompileArgs): void;
    /**
     * Check trainable weights count consistency.
     *
     * This will raise a warning if `this.trainableWeights` and
     * `this.collectedTrainableWeights` are inconsistent (i.e., have different
     * numbers of parameters).
     * Inconsistency will typically arise when one modifies `model.trainable`
     * without calling `model.compile()` again.
     */
    protected checkTrainableWeightsConsistency(): void;
    /**
     * Returns the loss value & metrics values for the model in test mode.
     *
     * Loss and metrics are specified during `compile()`, which needs to happen
     * before calls to `evaluate()`.
     *
     * Computation is done in batches.
     *
     * ```js
     * const model = tf.sequential({
     *   layers: [tf.layers.dense({units: 1, inputShape: [10]})]
     * });
     * model.compile({optimizer: 'sgd', loss: 'meanSquaredError'});
     * const result = model.evaluate(
     *     tf.ones([8, 10]), tf.ones([8, 1]), {batchSize: 4});
     * result.print();
     * ```
     *
     * @param x `tf.Tensor` of test data, or an `Array` of `tf.Tensor`s if the
     * model has multiple inputs.
     * @param y `tf.Tensor` of target data, or an `Array` of `tf.Tensor`s if the
     * model has multiple outputs.
     * @param args A `ModelEvaluateArgs`, containing optional fields.
     *
     * @return `Scalar` test loss (if the model has a single output and no
     *   metrics) or `Array` of `Scalar`s (if the model has multiple outputs
     *   and/or metrics). The attribute `model.metricsNames`
     *   will give you the display labels for the scalar outputs.
     *
     * @doc {heading: 'Models', subheading: 'Classes'}
     */
    evaluate(x: Tensor | Tensor[], y: Tensor | Tensor[], args?: ModelEvaluateArgs): Scalar | Scalar[];
    /**
     * Evaluate model using a dataset object.
     *
     * Note: Unlike `evaluate()`, this method is asynchronous (`async`).
     *
     * @param dataset A dataset object. Its `iterator()` method is expected
     *   to generate a dataset iterator object, the `next()` method of which
     *   is expected to produce data batches for evaluation. The return value
     *   of the `next()` call ought to contain a boolean `done` field and a
     *   `value` field. The `value` field is expected to be an array of two
     *   `tf.Tensor`s or an array of two nested `tf.Tensor` structures. The former
     *   case is for models with exactly one input and one output (e.g.
     *   a sequential model). The latter case is for models with multiple
     *   inputs and/or multiple outputs. Of the two items in the array, the
     *   first is the input feature(s) and the second is the output target(s).
     * @param args A configuration object for the dataset-based evaluation.
     * @returns Loss and metric values as an Array of `Scalar` objects.
     *
     * @doc {heading: 'Models', subheading: 'Classes'}
     */
    evaluateDataset(dataset: Dataset<{}>, args?: ModelEvaluateDatasetArgs): Promise<Scalar | Scalar[]>;
    /**
     * Get number of samples provided for training, evaluation or prediction.
     *
     * @param ins Input `tf.Tensor`.
     * @param batchSize Integer batch size, optional.
     * @param steps Total number of steps (batches of samples) before
     * declaring loop finished. Optional.
     * @param stepsName The public API's parameter name for `steps`.
     * @returns Number of samples provided.
     */
    private checkNumSamples;
    /**
     * Execute internal tensors of the model with input data feed.
     * @param inputs Input data feed. Must match the inputs of the model.
     * @param outputs Names of the output tensors to be fetched. Must match
     *   names of the SymbolicTensors that belong to the graph.
     * @returns Fetched values for `outputs`.
     */
    execute(inputs: Tensor | Tensor[] | NamedTensorMap, outputs: string | string[]): Tensor | Tensor[];
    /**
     * Retrieve the model's internal symbolic tensors from symbolic-tensor names.
     */
    private retrieveSymbolicTensors;
    /**
     * Helper method to loop over some data in batches.
     *
     * Porting Note: Not using the functional approach in the Python equivalent
     *   due to the imperative backend.
     * Porting Note: Does not support step mode currently.
     *
     * @param ins: input data
     * @param batchSize: integer batch size.
     * @param verbose: verbosity model
     * @returns: Predictions as `tf.Tensor` (if a single output) or an `Array` of
     *   `tf.Tensor` (if multipe outputs).
     */
    private predictLoop;
    /**
     * Generates output predictions for the input samples.
     *
     * Computation is done in batches.
     *
     * Note: the "step" mode of predict() is currently not supported.
     *   This is because the TensorFlow.js core backend is imperative only.
     *
     * ```js
     * const model = tf.sequential({
     *   layers: [tf.layers.dense({units: 1, inputShape: [10]})]
     * });
     * model.predict(tf.ones([8, 10]), {batchSize: 4}).print();
     * ```
     *
     * @param x The input data, as a Tensor, or an `Array` of `tf.Tensor`s if
     *   the model has multiple inputs.
     * @param args A `ModelPredictArgs` object containing optional fields.
     *
     * @return Prediction results as a `tf.Tensor`(s).
     *
     * @exception ValueError In case of mismatch between the provided input data
     *   and the model's expectations, or in case a stateful model receives a
     *   number of samples that is not a multiple of the batch size.
     *
     * @doc {heading: 'Models', subheading: 'Classes'}
     */
    predict(x: Tensor | Tensor[], args?: ModelPredictArgs): Tensor | Tensor[];
    /**
     * Returns predictions for a single batch of samples.
     *
     * ```js
     * const model = tf.sequential({
     *   layers: [tf.layers.dense({units: 1, inputShape: [10]})]
     * });
     * model.predictOnBatch(tf.ones([8, 10])).print();
     * ```
     * @param x: Input samples, as a Tensor (for models with exactly one
     *   input) or an array of Tensors (for models with more than one input).
     * @return Tensor(s) of predictions
     *
     * @doc {heading: 'Models', subheading: 'Classes'}
     */
    predictOnBatch(x: Tensor | Tensor[]): Tensor | Tensor[];
    protected standardizeUserDataXY(x: Tensor | Tensor[] | {
        [inputName: string]: Tensor;
    }, y: Tensor | Tensor[] | {
        [inputName: string]: Tensor;
    }, checkBatchAxis?: boolean, batchSize?: number): [Tensor[], Tensor[]];
    protected standardizeUserData(x: Tensor | Tensor[] | {
        [inputName: string]: Tensor;
    }, y: Tensor | Tensor[] | {
        [inputName: string]: Tensor;
    }, sampleWeight?: Tensor | Tensor[] | {
        [outputName: string]: Tensor;
    }, classWeight?: ClassWeight | ClassWeight[] | ClassWeightMap, checkBatchAxis?: boolean, batchSize?: number): Promise<[Tensor[], Tensor[], Tensor[]]>;
    /**
     * Loop over some test data in batches.
     * @param f A Function returning a list of tensors.
     * @param ins Array of tensors to be fed to `f`.
     * @param batchSize Integer batch size or `null` / `undefined`.
     * @param verbose verbosity mode.
     * @param steps Total number of steps (batches of samples) before
     * declaring test finished. Ignored with the default value of `null` /
     * `undefined`.
     * @returns Array of Scalars.
     */
    private testLoop;
    protected getDedupedMetricsNames(): string[];
    /**
     * Creates a function that performs the following actions:
     *
     * 1. computes the losses
     * 2. sums them to get the total loss
     * 3. call the optimizer computes the gradients of the LayersModel's
     *    trainable weights w.r.t. the total loss and update the variables
     * 4. calculates the metrics
     * 5. returns the values of the losses and metrics.
     */
    protected makeTrainFunction(): (data: Tensor[]) => Scalar[];
    /**
     * Create a function which, when invoked with an array of `tf.Tensor`s as a
     * batch of inputs, returns the prespecified loss and metrics of the model
     * under the batch of input data.
     */
    private makeTestFunction;
    /**
     * Trains the model for a fixed number of epochs (iterations on a
     * dataset).
     *
     * ```js
     * const model = tf.sequential({
     *     layers: [tf.layers.dense({units: 1, inputShape: [10]})]
     * });
     * model.compile({optimizer: 'sgd', loss: 'meanSquaredError'});
     * for (let i = 1; i < 5 ; ++i) {
     *   const h = await model.fit(tf.ones([8, 10]), tf.ones([8, 1]), {
     *       batchSize: 4,
     *       epochs: 3
     *   });
     *   console.log("Loss after Epoch " + i + " : " + h.history.loss[0]);
     * }
     * ```
     *
     * @param x `tf.Tensor` of training data, or an array of `tf.Tensor`s if the
     * model has multiple inputs. If all inputs in the model are named, you
     * can also pass a dictionary mapping input names to `tf.Tensor`s.
     * @param y `tf.Tensor` of target (label) data, or an array of `tf.Tensor`s if
     * the model has multiple outputs. If all outputs in the model are named,
     * you can also pass a dictionary mapping output names to `tf.Tensor`s.
     * @param args A `ModelFitArgs`, containing optional fields.
     *
     * @return A `History` instance. Its `history` attribute contains all
     *   information collected during training.
     *
     * @exception ValueError In case of mismatch between the provided input
     * data and what the model expects.
     *
     * @doc {heading: 'Models', subheading: 'Classes'}
     */
    fit(x: Tensor | Tensor[] | {
        [inputName: string]: Tensor;
    }, y: Tensor | Tensor[] | {
        [inputName: string]: Tensor;
    }, args?: ModelFitArgs): Promise<History>;
    /**
     * Abstract fit function for `f(ins)`.
     * @param f A Function returning a list of tensors. For training, this
     *   function is expected to perform the updates to the variables.
     * @param ins List of tensors to be fed to `f`.
     * @param outLabels List of strings, display names of the outputs of `f`.
     * @param batchSize Integer batch size or `== null` if unknown. Default : 32.
     * @param epochs Number of times to iterate over the data. Default : 1.
     * @param verbose Verbosity mode: 0, 1, or 2. Default: 1.
     * @param callbacks List of callbacks to be called during training.
     * @param valF Function to call for validation.
     * @param valIns List of tensors to be fed to `valF`.
     * @param shuffle Whether to shuffle the data at the beginning of every
     * epoch. Default : true.
     * @param callbackMetrics List of strings, the display names of the metrics
     *   passed to the callbacks. They should be the concatenation of the
     *   display names of the outputs of `f` and the list of display names
     *   of the outputs of `valF`.
     * @param initialEpoch Epoch at which to start training (useful for
     *   resuming a previous training run). Default : 0.
     * @param stepsPerEpoch Total number of steps (batches on samples) before
     *   declaring one epoch finished and starting the next epoch. Ignored with
     *   the default value of `undefined` or `null`.
     * @param validationSteps Number of steps to run validation for (only if
     *   doing validation from data tensors). Not applicable for tfjs-layers.
     * @returns A `History` object.
     */
    fitLoop(f: (data: Tensor[]) => Scalar[], ins: Tensor[], outLabels?: string[], batchSize?: number, epochs?: number, verbose?: number, callbacks?: BaseCallback[], valF?: (data: Tensor[]) => Scalar[], valIns?: Tensor[], shuffle?: boolean | string, callbackMetrics?: string[], initialEpoch?: number, stepsPerEpoch?: number, validationSteps?: number): Promise<History>;
    /**
     * Trains the model using a dataset object.
     *
     * @param dataset A dataset object. Its `iterator()` method is expected
     *   to generate a dataset iterator object, the `next()` method of which
     *   is expected to produce data batches for training. The return value
     *   of the `next()` call ought to contain a boolean `done` field and a
     *   `value` field. The `value` field is expected to be an array of two
     *   `tf.Tensor`s or an array of two nested `tf.Tensor` structures. The former
     *   case is for models with exactly one input and one output (e.g.
     *   a sequential model). The latter case is for models with multiple
     *   inputs and/or multiple outputs.
     *   Of the two items in the array, the first is the input feature(s) and
     *   the second is the output target(s).
     * @param args A `ModelFitDatasetArgs`, containing optional fields.
     *
     * @return A `History` instance. Its `history` attribute contains all
     *   information collected during training.
     *
     * @doc {heading: 'Models', subheading: 'Classes'}
     */
    fitDataset<T>(dataset: Dataset<T>, args: ModelFitDatasetArgs<T>): Promise<History>;
    /**
     * Runs a single gradient update on a single batch of data.
     *
     * This method differs from `fit()` and `fitDataset()` in the following
     * regards:
     *   - It operates on exactly one batch of data.
     *   - It returns only the loss and metric values, instead of
     *     returning the batch-by-batch loss and metric values.
     *   - It doesn't support fine-grained options such as verbosity and
     *     callbacks.
     *
     * @param x Input data. It could be one of the following:
     *   - A `tf.Tensor`, or an Array of `tf.Tensor`s (in case the model has
     *     multiple inputs).
     *   - An Object mapping input names to corresponding `tf.Tensor` (if the
     *     model has named inputs).
     * @param y Target data. It could be either a `tf.Tensor` or multiple
     *   `tf.Tensor`s. It should be consistent with `x`.
     * @returns Training loss or losses (in case the model has
     *   multiple outputs), along with metrics (if any), as numbers.
     *
     * @doc {heading: 'Models', subheading: 'Classes'}
     */
    trainOnBatch(x: Tensor | Tensor[] | {
        [inputName: string]: Tensor;
    }, y: Tensor | Tensor[] | {
        [inputName: string]: Tensor;
    }): Promise<number | number[]>;
    /**
     * Extract weight values of the model.
     *
     * @param config: An instance of `io.SaveConfig`, which specifies
     * model-saving options such as whether only trainable weights are to be
     * saved.
     * @returns A `NamedTensorMap` mapping original weight names (i.e.,
     *   non-uniqueified weight names) to their values.
     */
    protected getNamedWeights(config?: io.SaveConfig): NamedTensor[];
    /**
     * Setter used for force stopping of LayersModel.fit() (i.e., training).
     *
     * Example:
     *
     * ```js
     * const input = tf.input({shape: [10]});
     * const output = tf.layers.dense({units: 1}).apply(input);
     * const model = tf.model({inputs: [input], outputs: [output]});
     * model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});
     * const xs = tf.ones([8, 10]);
     * const ys = tf.zeros([8, 1]);
     *
     * const history = await model.fit(xs, ys, {
     *   epochs: 10,
     *   callbacks: {
     *     onEpochEnd: async (epoch, logs) => {
     *       if (epoch === 2) {
     *         model.stopTraining = true;
     *       }
     *     }
     *   }
     * });
     *
     * // There should be only 3 values in the loss array, instead of 10
     * values,
     * // due to the stopping after 3 epochs.
     * console.log(history.history.loss);
     * ```
     */
    set stopTraining(stop: boolean);
    get stopTraining(): boolean;
    get optimizer(): Optimizer;
    set optimizer(optimizer: Optimizer);
    dispose(): DisposeResult;
    private getLossIdentifiers;
    private getMetricIdentifiers;
    protected getTrainingConfig(): TrainingConfig;
    loadTrainingConfig(trainingConfig: TrainingConfig): void;
    /**
     * Save the configuration and/or weights of the LayersModel.
     *
     * An `IOHandler` is an object that has a `save` method of the proper
     * signature defined. The `save` method manages the storing or
     * transmission of serialized data ("artifacts") that represent the
     * model's topology and weights onto or via a specific medium, such as
     * file downloads, local storage, IndexedDB in the web browser and HTTP
     * requests to a server. TensorFlow.js provides `IOHandler`
     * implementations for a number of frequently used saving mediums, such as
     * `tf.io.browserDownloads` and `tf.io.browserLocalStorage`. See `tf.io`
     * for more details.
     *
     * This method also allows you to refer to certain types of `IOHandler`s
     * as URL-like string shortcuts, such as 'localstorage://' and
     * 'indexeddb://'.
     *
     * Example 1: Save `model`'s topology and weights to browser [local
     * storage](https://developer.mozilla.org/en-US/docs/Web/API/Window/localStorage);
     * then load it back.
     *
     * ```js
     * const model = tf.sequential(
     *     {layers: [tf.layers.dense({units: 1, inputShape: [3]})]});
     * console.log('Prediction from original model:');
     * model.predict(tf.ones([1, 3])).print();
     *
     * const saveResults = await model.save('localstorage://my-model-1');
     *
     * const loadedModel = await tf.loadLayersModel('localstorage://my-model-1');
     * console.log('Prediction from loaded model:');
     * loadedModel.predict(tf.ones([1, 3])).print();
     * ```
     *
     * Example 2. Saving `model`'s topology and weights to browser
     * [IndexedDB](https://developer.mozilla.org/en-US/docs/Web/API/IndexedDB_API);
     * then load it back.
     *
     * ```js
     * const model = tf.sequential(
     *     {layers: [tf.layers.dense({units: 1, inputShape: [3]})]});
     * console.log('Prediction from original model:');
     * model.predict(tf.ones([1, 3])).print();
     *
     * const saveResults = await model.save('indexeddb://my-model-1');
     *
     * const loadedModel = await tf.loadLayersModel('indexeddb://my-model-1');
     * console.log('Prediction from loaded model:');
     * loadedModel.predict(tf.ones([1, 3])).print();
     * ```
     *
     * Example 3. Saving `model`'s topology and weights as two files
     * (`my-model-1.json` and `my-model-1.weights.bin`) downloaded from
     * browser.
     *
     * ```js
     * const model = tf.sequential(
     *     {layers: [tf.layers.dense({units: 1, inputShape: [3]})]});
     * const saveResults = await model.save('downloads://my-model-1');
     * ```
     *
     * Example 4. Send  `model`'s topology and weights to an HTTP server.
     * See the documentation of `tf.io.http` for more details
     * including specifying request parameters and implementation of the
     * server.
     *
     * ```js
     * const model = tf.sequential(
     *     {layers: [tf.layers.dense({units: 1, inputShape: [3]})]});
     * const saveResults = await model.save('http://my-server/model/upload');
     * ```
     *
     * @param handlerOrURL An instance of `IOHandler` or a URL-like,
     * scheme-based string shortcut for `IOHandler`.
     * @param config Options for saving the model.
     * @returns A `Promise` of `SaveResult`, which summarizes the result of
     * the saving, such as byte sizes of the saved artifacts for the model's
     *   topology and weight values.
     *
     * @doc {heading: 'Models', subheading: 'Classes', ignoreCI: true}
     */
    save(handlerOrURL: io.IOHandler | string, config?: io.SaveConfig): Promise<io.SaveResult>;
    /**
     * Set user-defined metadata.
     *
     * The set metadata will be serialized together with the topology
     * and weights of the model during `save()` calls.
     *
     * @param setUserDefinedMetadata
     */
    setUserDefinedMetadata(userDefinedMetadata: {}): void;
    /**
     * Get user-defined metadata.
     *
     * The metadata is supplied via one of the two routes:
     *   1. By calling `setUserDefinedMetadata()`.
     *   2. Loaded during model loading (if the model is constructed
     *      via `tf.loadLayersModel()`.)
     *
     * If no user-defined metadata is available from either of the
     * two routes, this function will return `undefined`.
     */
    getUserDefinedMetadata(): {};
}
/**
 * A `tf.Functional` is an alias to `tf.LayersModel`.
 *
 * See also:
 *   `tf.LayersModel`, `tf.Sequential`, `tf.loadLayersModel`.
 */
/** @doc {heading: 'Models', subheading: 'Classes'} */
declare class Functional extends LayersModel {
    static className: string;
}
/// <amd-module name="@tensorflow/tfjs-layers/dist/keras_format/training_config" />

declare type MetricsIdentifier = string;
/**
 * a type for valid values of the `loss_weights` field.
 */
declare type LossWeights = number[] | {
    [key: string]: number;
};
/**
 * Configuration of the Keras trainer. This includes the configuration to the
 * optimizer, the loss, any metrics to be calculated, etc.
 */
interface TrainingConfig extends PyJsonDict {
    optimizer_config: OptimizerSerialization;
    loss: LossIdentifier | LossIdentifier[] | {
        [key: string]: LossIdentifier;
    };
    metrics?: MetricsIdentifier[] | {
        [key: string]: MetricsIdentifier;
    };
    weighted_metrics?: MetricsIdentifier[];
    sample_weight_mode?: SampleWeightMode;
    loss_weights?: LossWeights;
}

/// <amd-module name="@tensorflow/tfjs-layers/dist/engine/training_dataset" />
/**
 * Interfaces and methods for training models using TensorFlow.js datasets.
 */
/**
 * Interface for configuring model training based on a dataset object.
 */
interface ModelFitDatasetArgs<T> {
    /**
     * (Optional) Total number of steps (batches of samples) before
     * declaring one epoch finished and starting the next epoch. It should
     * typically be equal to the number of samples of your dataset divided by
     * the batch size, so that `fitDataset`() call can utilize the entire dataset.
     * If it is not provided, use `done` return value in `iterator.next()` as
     * signal to finish an epoch.
     */
    batchesPerEpoch?: number;
    /**
     * Integer number of times to iterate over the training dataset.
     */
    epochs: number;
    /**
     * Verbosity level.
     *
     * Expected to be 0, 1, or 2. Default: 1.
     *
     * 0 - No printed message during fit() call.
     * 1 - In Node.js (tfjs-node), prints the progress bar, together with
     *     real-time updates of loss and metric values and training speed.
     *     In the browser: no action. This is the default.
     * 2 - Not implemented yet.
     */
    verbose?: ModelLoggingVerbosity;
    /**
     * List of callbacks to be called during training.
     * Can have one or more of the following callbacks:
     *   - `onTrainBegin(logs)`: called when training starts.
     *   - `onTrainEnd(logs)`: called when training ends.
     *   - `onEpochBegin(epoch, logs)`: called at the start of every epoch.
     *   - `onEpochEnd(epoch, logs)`: called at the end of every epoch.
     *   - `onBatchBegin(batch, logs)`: called at the start of every batch.
     *   - `onBatchEnd(batch, logs)`: called at the end of every batch.
     *   - `onYield(epoch, batch, logs)`: called every `yieldEvery` milliseconds
     *      with the current epoch, batch and logs. The logs are the same
     *      as in `onBatchEnd()`. Note that `onYield` can skip batches or
     *      epochs. See also docs for `yieldEvery` below.
     */
    callbacks?: BaseCallback[] | CustomCallbackArgs | CustomCallbackArgs[];
    /**
     * Data on which to evaluate the loss and any model
     * metrics at the end of each epoch. The model will not be trained on this
     * data. This could be any of the following:
     *
     *   - An array `[xVal, yVal]`, where the two values may be `tf.Tensor`,
     *     an array of Tensors, or a map of string to Tensor.
     *   - Similarly, an array ` [xVal, yVal, valSampleWeights]`
     *     (not implemented yet).
     *   - a `Dataset` object with elements of the form `{xs: xVal, ys: yVal}`,
     *     where `xs` and `ys` are the feature and label tensors, respectively.
     *
     * If `validationData` is an Array of Tensor objects, each `tf.Tensor` will be
     * sliced into batches during validation, using the parameter
     * `validationBatchSize` (which defaults to 32). The entirety of the
     * `tf.Tensor` objects will be used in the validation.
     *
     * If `validationData` is a dataset object, and the `validationBatches`
     * parameter is specified, the validation will use `validationBatches` batches
     * drawn from the dataset object. If `validationBatches` parameter is not
     * specified, the validation will stop when the dataset is exhausted.
     *
     * The model will not be trained on this data.
     */
    validationData?: [
        TensorOrArrayOrMap,
        TensorOrArrayOrMap
    ] | [TensorOrArrayOrMap, TensorOrArrayOrMap, TensorOrArrayOrMap] | Dataset<T>;
    /**
     * Optional batch size for validation.
     *
     * Used only if `validationData` is an array of `tf.Tensor` objects, i.e., not
     * a dataset object.
     *
     * If not specified, its value defaults to 32.
     */
    validationBatchSize?: number;
    /**
     * (Optional) Only relevant if `validationData` is specified and is a dataset
     * object.
     *
     * Total number of batches of samples to draw from `validationData` for
     * validation purpose before stopping at the end of every epoch. If not
     * specified, `evaluateDataset` will use `iterator.next().done` as signal to
     * stop validation.
     */
    validationBatches?: number;
    /**
     * Configures the frequency of yielding the main thread to other tasks.
     *
     * In the browser environment, yielding the main thread can improve the
     * responsiveness of the page during training. In the Node.js environment,
     * it can ensure tasks queued in the event loop can be handled in a timely
     * manner.
     *
     * The value can be one of the following:
     *   - `'auto'`: The yielding happens at a certain frame rate (currently set
     *               at 125ms). This is the default.
     *   - `'batch'`: yield every batch.
     *   - `'epoch'`: yield every epoch.
     *   - a `number`: Will yield every `number` milliseconds.
     *   - `'never'`: never yield. (But yielding can still happen through `await
     *      nextFrame()` calls in custom callbacks.)
     */
    yieldEvery?: YieldEveryOptions;
    /**
     * Epoch at which to start training (useful for resuming a previous training
     * run). When this is used, `epochs` is the index of the "final epoch".
     * The model is not trained for a number of iterations given by `epochs`,
     * but merely until the epoch of index `epochs` is reached.
     */
    initialEpoch?: number;
    /**
     * Optional object mapping class indices (integers) to
     * a weight (float) to apply to the model's loss for the samples from this
     * class during training. This can be useful to tell the model to "pay more
     * attention" to samples from an under-represented class.
     *
     * If the model has multiple outputs, a class weight can be specified for
     * each of the outputs by setting this field an array of weight object
     * or an object that maps model output names (e.g., `model.outputNames[0]`)
     * to weight objects.
     */
    classWeight?: ClassWeight | ClassWeight[] | ClassWeightMap;
}
interface FitDatasetElement {
    xs: TensorOrArrayOrMap;
    ys: TensorOrArrayOrMap;
}
/**
 * Interface for configuring model evaluation based on a dataset object.
 */
interface ModelEvaluateDatasetArgs {
    /**
     * Number of batches to draw from the dataset object before ending the
     * evaluation.
     */
    batches?: number;
    /**
     * Verbosity mode.
     */
    verbose?: ModelLoggingVerbosity;
}
declare function fitDataset<T>(model: any, dataset: Dataset<T>, args: ModelFitDatasetArgs<T>): Promise<History>;
declare function evaluateDataset<T>(model: any, dataset: Dataset<T> | LazyIterator<T>, args: ModelEvaluateDatasetArgs): Promise<tfc.Scalar | tfc.Scalar[]>;

/// <amd-module name="@tensorflow/tfjs-layers/dist/engine/training_tensors" />
/**
 * Interface configuration model training based on data as `tf.Tensor`s.
 */
interface ModelFitArgs {
    /**
     * Number of samples per gradient update. If unspecified, it
     * will default to 32.
     */
    batchSize?: number;
    /**
     * Integer number of times to iterate over the training data arrays.
     */
    epochs?: number;
    /**
     * Verbosity level.
     *
     * Expected to be 0, 1, or 2. Default: 1.
     *
     * 0 - No printed message during fit() call.
     * 1 - In Node.js (tfjs-node), prints the progress bar, together with
     *     real-time updates of loss and metric values and training speed.
     *     In the browser: no action. This is the default.
     * 2 - Not implemented yet.
     */
    verbose?: ModelLoggingVerbosity;
    /**
     * List of callbacks to be called during training.
     * Can have one or more of the following callbacks:
     *   - `onTrainBegin(logs)`: called when training starts.
     *   - `onTrainEnd(logs)`: called when training ends.
     *   - `onEpochBegin(epoch, logs)`: called at the start of every epoch.
     *   - `onEpochEnd(epoch, logs)`: called at the end of every epoch.
     *   - `onBatchBegin(batch, logs)`: called at the start of every batch.
     *   - `onBatchEnd(batch, logs)`: called at the end of every batch.
     *   - `onYield(epoch, batch, logs)`: called every `yieldEvery` milliseconds
     *      with the current epoch, batch and logs. The logs are the same
     *      as in `onBatchEnd()`. Note that `onYield` can skip batches or
     *      epochs. See also docs for `yieldEvery` below.
     */
    callbacks?: BaseCallback[] | CustomCallbackArgs | CustomCallbackArgs[];
    /**
     * Float between 0 and 1: fraction of the training data
     * to be used as validation data. The model will set apart this fraction of
     * the training data, will not train on it, and will evaluate the loss and
     * any model metrics on this data at the end of each epoch.
     * The validation data is selected from the last samples in the `x` and `y`
     * data provided, before shuffling.
     */
    validationSplit?: number;
    /**
     * Data on which to evaluate the loss and any model
     * metrics at the end of each epoch. The model will not be trained on this
     * data. This could be a tuple [xVal, yVal] or a tuple [xVal, yVal,
     * valSampleWeights]. The model will not be trained on this data.
     * `validationData` will override `validationSplit`.
     */
    validationData?: [
        Tensor | Tensor[],
        Tensor | Tensor[]
    ] | [Tensor | Tensor[], Tensor | Tensor[], Tensor | Tensor[]];
    /**
     * Whether to shuffle the training data before each epoch. Has
     * no effect when `stepsPerEpoch` is not `null`.
     */
    shuffle?: boolean;
    /**
     * Optional object mapping class indices (integers) to
     * a weight (float) to apply to the model's loss for the samples from this
     * class during training. This can be useful to tell the model to "pay more
     * attention" to samples from an under-represented class.
     *
     * If the model has multiple outputs, a class weight can be specified for
     * each of the outputs by setting this field an array of weight object
     * or an object that maps model output names (e.g., `model.outputNames[0]`)
     * to weight objects.
     */
    classWeight?: ClassWeight | ClassWeight[] | ClassWeightMap;
    /**
     * Optional array of the same length as x, containing
     * weights to apply to the model's loss for each sample. In the case of
     * temporal data, you can pass a 2D array with shape (samples,
     * sequenceLength), to apply a different weight to every timestep of every
     * sample. In this case you should make sure to specify
     * sampleWeightMode="temporal" in compile().
     */
    sampleWeight?: Tensor;
    /**
     * Epoch at which to start training (useful for resuming a previous training
     * run). When this is used, `epochs` is the index of the "final epoch".
     * The model is not trained for a number of iterations given by `epochs`,
     * but merely until the epoch of index `epochs` is reached.
     */
    initialEpoch?: number;
    /**
     * Total number of steps (batches of samples) before
     * declaring one epoch finished and starting the next epoch. When training
     * with Input Tensors such as TensorFlow data tensors, the default `null` is
     * equal to the number of unique samples in your dataset divided by the
     * batch size, or 1 if that cannot be determined.
     */
    stepsPerEpoch?: number;
    /**
     * Only relevant if `stepsPerEpoch` is specified. Total number of steps
     * (batches of samples) to validate before stopping.
     */
    validationSteps?: number;
    /**
     * Configures the frequency of yielding the main thread to other tasks.
     *
     * In the browser environment, yielding the main thread can improve the
     * responsiveness of the page during training. In the Node.js environment,
     * it can ensure tasks queued in the event loop can be handled in a timely
     * manner.
     *
     * The value can be one of the following:
     *   - `'auto'`: The yielding happens at a certain frame rate (currently set
     *               at 125ms). This is the default.
     *   - `'batch'`: yield every batch.
     *   - `'epoch'`: yield every epoch.
     *   - any `number`: yield every `number` milliseconds.
     *   - `'never'`: never yield. (yielding can still happen through `await
     *      nextFrame()` calls in custom callbacks.)
     */
    yieldEvery?: YieldEveryOptions;
}
declare function checkBatchSize(batchSize: number): void;
/**
 * Slice a Tensor or an Array of Tensors, by start and stop indices.
 *
 * Porting Note: The `_slice_arrays` function in PyKeras is covered by this
 *   function and `sliceArraysByIndices()` together.
 *
 * @param arrays: the input.
 * @param start: the starting index (inclusive).
 * @param stop: the stopping index (exclusive).
 * @returns The result of the slicing. If `arrays` is an `Array` of
 *   `tf.Tensor`s, the slicing will be applied to all elements of the `Array`
 *   in the same way.
 */
declare function sliceArrays(arrays: Tensor | Tensor[], start: number, stop: number): Tensor | Tensor[];
/**
 * Slice a Tensor or an Array of Tensors, by random-order indices.
 *
 * Porting Note: The `_slice_arrays` function in PyKeras is covered by this
 *   function and `sliceArrays()` together.
 *
 * @param arrays The input `tf.Tensor` or `Array` of `tf.Tensor`s to slice.
 *   If an `Array` of `tf.Tensor`s, all `tf.Tensor`s will be sliced in the
 *   same fashion.
 * @param indices The indices to use for slicing along the first (batch)
 *   dimension.
 * @returns Result(s) of the slicing.
 */
declare function sliceArraysByIndices(arrays: Tensor | Tensor[], indices: Tensor1D): Tensor | Tensor[];
/**
 * Returns a list of batch indices (tuples of indices).
 * @param size: Integer, total size of the data to slice into batches.
 * @param batchSize: Integer, batch size.
 * @returns An Array of [batchStart, batchEnd] tuples. batchStart is
 *   inclusive; batchEnd is exclusive. I.e., each batch consists of indices x
 *   that satisfy batchStart <= x < batchEnd.
 */
declare function makeBatches(size: number, batchSize: number): Array<[number, number]>;
/**
 * Ensure tensors all have a rank of at least 2.
 *
 * If a tensor has a rank of 1, it is dimension-expanded to rank 2.
 * If any tensor has a rank of 0 (i.e., is a scalar), an error will be thrown.
 */
declare function ensureTensorsRank2OrHigher(tensors: Tensor | Tensor[]): Tensor[];
/**
 * Compare a set of tensors with a reference (old) set, discard the ones
 * in the new set that are not present in the reference set.
 *
 * This method is used for memory clenaup during calls such as
 * LayersModel.fit().
 *
 * @param tensors New set which may contain Tensors not present in
 *   `refTensors`.
 * @param refTensors Reference Tensor set.
 */
declare function disposeNewTensors(tensors: Tensor | Tensor[] | {
    [inputName: string]: Tensor;
}, refTensors: Tensor | Tensor[] | {
    [inputName: string]: Tensor;
}): void;

/// <amd-module name="@tensorflow/tfjs-layers/dist/engine/training_utils" />
/**
 * For multi-class classification problems, this object is designed to store a
 * mapping from class index to the "weight" of the class, where higher weighted
 * classes have larger impact on loss, accuracy, and other metrics.
 *
 * This is useful for cases in which you want the model to "pay more attention"
 * to examples from an under-represented class, e.g., in unbalanced datasets.
 */
declare type ClassWeight = {
    [classIndex: number]: number;
};
/**
 * Class weighting for a model with multiple outputs.
 *
 * This object maps each output name to a class-weighting object.
 */
declare type ClassWeightMap = {
    [outputName: string]: ClassWeight;
};
/**
 * Standardize class weighting objects.
 *
 * This function takes a single class-weighting object, an array of them,
 * or a map from output name to class-weighting object. It compares it to the
 * output name(s) of the model, base on which it outputs an array of
 * class-weighting objects of which the length matches the number of outputs.
 *
 * @param classWeight Input class-weighting object(s).
 * @param outputNames All output name(s) of the model.
 * @return An array of class-weighting objects. The length of the array matches
 *   the model's number of outputs.
 */
declare function standardizeClassWeights(classWeight: ClassWeight | ClassWeight[] | ClassWeightMap, outputNames: string[]): ClassWeight[];
declare function standardizeSampleWeights(classWeight: ClassWeight | ClassWeight[] | ClassWeightMap, outputNames: string[]): ClassWeight[];
/**
 * Standardize by-sample and/or by-class weights for training.
 *
 * Note that this function operates on one model output at a time. For a model
 * with multiple outputs, you must call this function multiple times.
 *
 * @param y The target tensor that the by-sample and/or by-class weight is for.
 *     The values of y are assumed to encode the classes, either directly
 *     as an integer index, or as one-hot encoding.
 * @param sampleWeight By-sample weights.
 * @param classWeight By-class weights: an object mapping class indices
 *     (integers) to a weight (float) to apply to the model's loss for the
 *     samples from this class during training. This can be useful to tell the
 *     model to "pay more attention" to samples from an under-represented class.
 * @param sampleWeightMode The mode for the sample weights.
 * @return A Promise of weight tensor, of which the size of the first dimension
 *     matches that of `y`.
 */
declare function standardizeWeights(y: Tensor, sampleWeight?: Tensor, classWeight?: ClassWeight, sampleWeightMode?: 'temporal'): Promise<Tensor>;
/**
 * Apply per-sample weights on the loss values from a number of samples.
 *
 * @param losses Loss tensor of shape `[batchSize]`.
 * @param sampleWeights Per-sample weight tensor of shape `[batchSize]`.
 * @returns Tensor of the same shape as`losses`.
 */
declare function computeWeightedLoss(losses: Tensor, sampleWeights: Tensor): Tensor<import("@tensorflow/tfjs-core").Rank>;
/// <amd-module name="@tensorflow/tfjs-core/dist/ops/image/transform" />
/**
 * Applies the given transform(s) to the image(s).
 *
 * @param image 4d tensor of shape `[batch, imageHeight, imageWidth, depth]`.
 * @param transforms Projective transform matrix/matrices. A tensor1d of length
 *     8 or tensor of size N x 8. If one row of transforms is [a0, a1, a2, b0,
 *     b1, b2, c0, c1], then it maps the output point (x, y) to a transformed
 *     input point (x', y') = ((a0 x + a1 y + a2) / k, (b0 x + b1 y + b2) / k),
 *     where k = c0 x + c1 y + 1. The transforms are inverted compared to the
 *     transform mapping input points to output points.
 * @param interpolation Interpolation mode.
 *     Supported values: 'nearest', 'bilinear'. Default to 'nearest'.
 * @param fillMode Points outside the boundaries of the input are filled
 *     according to the given mode, one of 'constant', 'reflect', 'wrap',
 *     'nearest'. Default to 'constant'.
 *     'reflect': (d c b a | a b c d | d c b a ) The input is extended by
 *     reflecting about the edge of the last pixel.
 *     'constant': (k k k k | a b c d | k k k k) The input is extended by
 *     filling all values beyond the edge with the same constant value k.
 *     'wrap': (a b c d | a b c d | a b c d) The input is extended by
 *     wrapping around to the opposite edge.
 *     'nearest': (a a a a | a b c d | d d d d) The input is extended by
 *     the nearest pixel.
 * @param fillValue A float represents the value to be filled outside the
 *     boundaries when fillMode is 'constant'.
 * @param Output dimension after the transform, [height, width]. If undefined,
 *     output is the same size as input image.
 *
 * @doc {heading: 'Operations', subheading: 'Images', namespace: 'image'}
 */
declare function transform_(image: Tensor4D | TensorLike, transforms: Tensor2D | TensorLike, interpolation?: 'nearest' | 'bilinear', fillMode?: 'constant' | 'reflect' | 'wrap' | 'nearest', fillValue?: number, outputShape?: [number, number]): Tensor4D;
declare const transform: typeof transform_;
{ };
/// <amd-module name="@tensorflow/tfjs-core/dist/ops/image/transform_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/public/chained_ops/transpose" />
declare module '../../tensor' {
    interface Tensor<R extends Rank = Rank> {
        transpose<T extends Tensor>(perm?: number[]): T;
    }
}

/// <amd-module name="@tensorflow/tfjs-core/dist/gradients/Transpose_grad" />
declare const transposeGradConfig: GradConfig;

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/transpose_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/truncated_normal" />
/**
 * Creates a `tf.Tensor` with values sampled from a truncated normal
 * distribution.
 *
 * ```js
 * tf.truncatedNormal([2, 2]).print();
 * ```
 *
 * The generated values follow a normal distribution with specified mean and
 * standard deviation, except that values whose magnitude is more than 2
 * standard deviations from the mean are dropped and re-picked.
 *
 * @param shape An array of integers defining the output tensor shape.
 * @param mean The mean of the normal distribution.
 * @param stdDev The standard deviation of the normal distribution.
 * @param dtype The data type of the output tensor.
 * @param seed The seed for the random number generator.
 *
 * @doc {heading: 'Tensors', subheading: 'Creation'}
 */
declare function truncatedNormal_<R extends Rank>(shape: ShapeMap[R], mean?: number, stdDev?: number, dtype?: 'float32' | 'int32', seed?: number): Tensor<R>;
declare const truncatedNormal: typeof truncatedNormal_;
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/truncated_normal_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-layers/dist/keras_format/types" />
/**
 * A value within the JSON-serialized form of a serializable object.
 *
 * The keys of any nested dicts should be in snake_case (i.e., using Python
 * naming conventions) for compatibility with Python Keras.
 *
 * @see PyJsonDict
 */
declare type PyJsonValue = boolean | number | string | null | PyJsonArray | PyJsonDict;
/**
 * A key-value dict within the JSON-serialized form of a serializable object.
 *
 * Serialization/deserialization uses stringified-JSON as the storage
 * representation. Typically this should be used for materialized JSON
 * stored on disk or sent/received over the wire.
 *
 * The keys of this dict and of any nested dicts should be in snake_case (i.e.,
 * using Python naming conventions) for compatibility with Python Keras.
 *
 * Internally this is normally converted to a ConfigDict that has CamelCase keys
 * (using TypeScript naming conventions) and support for Enums.
 */
interface PyJsonDict {
    [key: string]: PyJsonValue;
}
/**
 * A key-value dict like @see PyJsonDict, but with restricted keys.
 *
 * This makes it possible to create subtypes that have only the specified
 * fields, while requiring that the values are JSON-compatible.
 *
 * That is in contrast to extending `PyJsonDict`, or using an intersection type
 * `Foo & PyJsonDict`.  In both of those cases, the fields of Foo are actually
 * allowed to be of types that are incompatible with `PyJsonValue`.  Worse, the
 * index signature of `PyJsonValue` means that *any* key is accepted: eg.
 * `const foo: Foo = ...; foo.bogus = 12; const x = foo.bogus` works for both
 * reading and assignment, even if `bogus` is not a field of the type `Foo`,
 * because the index signature inherited from `PyJsonDict` accepts all strings.
 *
 * Here, we *both* restrict the keys to known values, *and* guarantee that the
 * values associated with those keys are compatible with `PyJsonValue`.
 *
 * This guarantee is easiest to apply via an additional incantation:
 *
 * ```
 * interface Foo extends PyJson<keyof Foo> {
 *   a: SomeType;
 *   b: SomeOtherType;
 * }
 * ```
 *
 * Now instances of `Foo` have *only* the fields `a` and `b`, and furthermore,
 * if either the type `SomeType` or `SomeOtherType` is incompatible with
 * `PyJsonValue`, the compiler produces a typing error.
 */
declare type PyJson<Keys extends string> = {
    [x in Keys]?: PyJsonValue;
};
/**
 * An array of values within the JSON-serialized form of a serializable object.
 *
 * The keys of any nested dicts should be in snake_case (i.e., using Python
 * naming conventions) for compatibility with Python Keras.
 *
 * @see PyJsonDict
 */
interface PyJsonArray extends Array<PyJsonValue> {
}
/**
 * A Keras JSON entry representing a Keras object such as a Layer.
 *
 * The Keras JSON convention is to provide the `class_name` (e.g., the layer
 * type) at the top level, and then to place the class-specific configuration in
 * a `config` subtree.  These class-specific configurations are provided by
 * subtypes of `PyJsonDict`.  Thus, this `*Serialization` has a type parameter
 * giving the specific type of the wrapped `PyJsonDict`.
 */
interface BaseSerialization<N extends string, T extends PyJson<Extract<keyof T, string>>> extends PyJsonDict {
    class_name: N;
    config: T;
}

/// <amd-module name="@tensorflow/tfjs-core/dist/types_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-layers/dist/utils/types_utils" />
/**
 * Determine whether the input is an Array of Shapes.
 */
declare function isArrayOfShapes(x: Shape | Shape[]): boolean;
/**
 * Special case of normalizing shapes to lists.
 *
 * @param x A shape or list of shapes to normalize into a list of Shapes.
 * @return A list of Shapes.
 */
declare function normalizeShapeList(x: Shape | Shape[]): Shape[];
/**
 * Helper function to obtain exactly one Tensor.
 * @param xs: A single `tf.Tensor` or an `Array` of `tf.Tensor`s.
 * @return A single `tf.Tensor`. If `xs` is an `Array`, return the first one.
 * @throws ValueError: If `xs` is an `Array` and its length is not 1.
 */
declare function getExactlyOneTensor(xs: Tensor | Tensor[]): Tensor;
/**
 * Helper function to obtain exactly on instance of Shape.
 *
 * @param shapes Input single `Shape` or Array of `Shape`s.
 * @returns If input is a single `Shape`, return it unchanged. If the input is
 *   an `Array` containing exactly one instance of `Shape`, return the instance.
 *   Otherwise, throw a `ValueError`.
 * @throws ValueError: If input is an `Array` of `Shape`s, and its length is not
 *   1.
 */
declare function getExactlyOneShape(shapes: Shape | Shape[]): Shape;

/// <amd-module name="@tensorflow/tfjs-core/dist/public/chained_ops/unique" />
declare module '../../tensor' {
    interface Tensor<R extends Rank = Rank> {
        unique<T extends Tensor>(this: T, axis?: number): {
            values: T;
            indices: T;
        };
    }
}

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/unique_test" />
{ };
/// <amd-module name="@tensorflow/tfjs-core/dist/gradients/Unpack_grad" />
declare const unpackGradConfig: GradConfig;

/// <amd-module name="@tensorflow/tfjs-core/dist/gradients/UnsortedSegmentSum_grad" />
declare const unsortedSegmentSumGradConfig: GradConfig;

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/unsorted_segment_sum" />
/**
 * Computes the sum along segments of a `tf.Tensor`.
 *
 * ```js
 * const x = tf.tensor1d([1, 2, 3, 4]);
 * const segmentIds = tf.tensor1d([1, 2, 0, 1], 'int32');
 * const numSegments = 3;
 *
 * x.unsortedSegmentSum(segmentIds, numSegments).print()
 * //or tf.unsortedSegmentSum(x, segmentIds, numSegments)
 * ```
 * @param x The `tf.Tensor` that will be summed along its segments.
 * @param segmentIds A `tf.Tensor1D` whose rank is equal to the rank of `x`'s
 * dimension along the `axis`.  Maps each element of `x` to a segment.
 * @param numSegments The number of distinct `segmentIds`.
 *
 * @doc {heading: 'Operations', subheading: 'Segment'}
 */
declare function unsortedSegmentSum_<T extends Tensor>(x: T | TensorLike, segmentIds: Tensor1D | TensorLike, numSegments: number): T;
declare const unsortedSegmentSum: typeof unsortedSegmentSum_;
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/unsorted_segment_sum_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/unstack" />
/**
 * Unstacks a `tf.Tensor` of rank-`R` into a list of rank-`(R-1)` `tf.Tensor`s.
 *
 * ```js
 * const a = tf.tensor2d([1, 2, 3, 4], [2, 2]);
 *
 * tf.unstack(a).forEach(tensor => tensor.print());
 * ```
 *
 * @param x A tensor object.
 * @param axis The axis to unstack along. Defaults to 0 (the first dim).
 *
 * @doc {heading: 'Tensors', subheading: 'Slicing and Joining'}
 */
declare function unstack_(x: Tensor | TensorLike, axis?: number): Tensor[];
declare const unstack: typeof unstack_;
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/unstack_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/upper_bound" />
/**
 * Searches for where a value would go in a sorted sequence.
 *
 * This is not a method for checking containment (like javascript in).
 *
 * The typical use case for this operation is "binning", "bucketing", or
 * "discretizing". The values are assigned to bucket-indices based on the edges
 * listed in 'sortedSequence'. This operation returns the bucket-index for each
 * value.
 *
 * The index returned corresponds to the first edge greater than the value.
 *
 * The axis is not settable for this operation. It always operates on the
 * innermost dimension (axis=-1). The operation will accept any number of outer
 * dimensions.
 *
 * Note: This operation assumes that 'upperBound' is sorted along the
 * innermost axis, maybe using 'sort(..., axis=-1)'. If the sequence is not
 * sorted no error is raised and the content of the returned tensor is not well
 * defined.
 *
 * ```js
 * const seq = tf.tensor1d([0, 3, 9, 10, 10]);
 * const values = tf.tensor1d([0, 4, 10]);
 * const result = tf.upperBound(seq, values);
 * result.print(); // [1, 2, 5]
 * ```
 * @param sortedSequence: N-D. Sorted sequence.
 * @param values: N-D. Search values.
 * @return An N-D int32 tensor the size of values containing the result of
 *     applying upper bound to each value. The result is not a global index to
 *     the entire Tensor, but the index in the last dimension.
 * @doc {heading: 'Operations', subheading: 'Evaluation'}
 */
declare function upperBound(sortedSequence: Tensor | TensorLike, values: Tensor | TensorLike): Tensor;

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/upper_bound_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-data/dist/iterators/url_chunk_iterator" />
/**
 * Provide a stream of chunks from a URL.
 *
 * Note this class first downloads the entire file into memory before providing
 * the first element from the stream.  This is because the Fetch API does not
 * yet reliably provide a reader stream for the response body.
 */
declare function urlChunkIterator(url: RequestInfo, options?: FileChunkIteratorOptions, fetchFunc?: Function): Promise<FileChunkIterator>;

/// <amd-module name="@tensorflow/tfjs-data/dist/sources/url_data_source" />
declare class URLDataSource extends DataSource {
    protected readonly url: RequestInfo;
    protected readonly fileOptions: FileChunkIteratorOptions;
    /**
     * Create a `URLDataSource`.
     *
     * @param url A source URL string, or a `Request` object.
     * @param options Options passed to the underlying `FileChunkIterator`s,
     *   such as {chunksize: 1024}.
     */
    constructor(url: RequestInfo, fileOptions?: FileChunkIteratorOptions);
    iterator(): Promise<ByteChunkIterator>;
}

/// <amd-module name="@tensorflow/tfjs-layers/dist/user_defined_metadata" />
/** Utility functions related to user-defined metadata. */
declare const MAX_USER_DEFINED_METADATA_SERIALIZED_LENGTH: number;
/**
 * Check validity of user-defined metadata.
 *
 * @param userDefinedMetadata
 * @param modelName Name of the model that the user-defined metadata belongs to.
 *   Used during construction of error messages.
 * @param checkSize Whether to check the size of the metadata is under
 *   recommended limit. Default: `false`. If `true`, will try stringify the
 *   JSON object and print a console warning if the serialzied size is above the
 *   limit.
 * @throws Error if `userDefinedMetadata` is not a plain JSON object.
 */
declare function checkUserDefinedMetadata(userDefinedMetadata: {}, modelName: string, checkSize?: boolean): void;
/**
 * Check if an input is plain JSON object or any valid subfield of it.
 *
 * @param x The input to be checked.
 * @param assertObject Whether to assert `x` is a JSON object, i.e., reject
 *   cases of arrays and primitives.
 * @return Returns `true` if and only if `x` is a plain JSON object,
 *   a JSON-valid primitive including string, number, boolean and null,
 *   or an array of the said types.
 */
declare function plainObjectCheck(x: any): boolean;

/// <amd-module name="@tensorflow/tfjs-core/dist/util" />
/**
 * Create typed array for scalar value. Used for storing in `DataStorage`.
 */
declare function createScalarValue(value: DataType, dtype: DataType): BackendValues;
declare function toTypedArray(a: TensorLike, dtype: DataType): TypedArray;
/**
 * Returns the current high-resolution time in milliseconds relative to an
 * arbitrary time in the past. It works across different platforms (node.js,
 * browsers).
 *
 * ```js
 * console.log(tf.util.now());
 * ```
 *
 * @doc {heading: 'Util', namespace: 'util'}
 */
declare function now(): number;
/**
 * Returns a platform-specific implementation of
 * [`fetch`](https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API).
 *
 * If `fetch` is defined on the global object (`window`, `process`, etc.),
 * `tf.util.fetch` returns that function.
 *
 * If not, `tf.util.fetch` returns a platform-specific solution.
 *
 * ```js
 * const resource = await tf.util.fetch('https://unpkg.com/@tensorflow/tfjs');
 * // handle response
 * ```
 *
 * @doc {heading: 'Util'}
 */
declare function fetch(path: string, requestInits?: RequestInit): Promise<Response>;
/**
 * Encodes the provided string into bytes using the provided encoding scheme.
 *
 * @param s The string to encode.
 * @param encoding The encoding scheme. Defaults to utf-8.
 *
 * @doc {heading: 'Util'}
 */
declare function encodeString(s: string, encoding?: string): Uint8Array;
/**
 * Decodes the provided bytes into a string using the provided encoding scheme.
 * @param bytes The bytes to decode.
 *
 * @param encoding The encoding scheme. Defaults to utf-8.
 *
 * @doc {heading: 'Util'}
 */
declare function decodeString(bytes: Uint8Array, encoding?: string): string;
declare function isTypedArray(a: {}): a is Float32Array | Int32Array | Uint8Array | Uint8ClampedArray;
/**
 *  Flattens an arbitrarily nested array.
 *
 * ```js
 * const a = [[1, 2], [3, 4], [5, [6, [7]]]];
 * const flat = tf.util.flatten(a);
 * console.log(flat);
 * ```
 *
 *  @param arr The nested array to flatten.
 *  @param result The destination array which holds the elements.
 *  @param skipTypedArray If true, avoids flattening the typed arrays. Defaults
 *      to false.
 *
 * @doc {heading: 'Util', namespace: 'util'}
 */
declare function flatten<T extends number | boolean | string | Promise<number> | TypedArray>(arr: T | RecursiveArray<T>, result?: T[], skipTypedArray?: boolean): T[];

/// <amd-module name="@tensorflow/tfjs-layers/dist/keras_format/utils" />
/**
 * Infers a string union type from an array of string literals, and returns
 * the array as an array of that type.
 *
 * For instance:
 *
 * ```
 * const fruits = stringLiteralArray(['apple', 'banana', 'orange']);
 * type Fruit = typeof activationOptions[number];
 * ```
 *
 * now `Fruit` is the union type `'apple'|'banana'|'orange'`.
 *
 * https://stackoverflow.com/questions/52085454/typescript-define-a-union-type-from-an-array-of-strings/52085658
 */
declare function stringLiteralArray<T extends string>(a: T[]): T[];

/// <amd-module name="@tensorflow/tfjs-core/dist/util_base" />
/**
 * Shuffles the array in-place using Fisher-Yates algorithm.
 *
 * ```js
 * const a = [1, 2, 3, 4, 5];
 * tf.util.shuffle(a);
 * console.log(a);
 * ```
 *
 * @param array The array to shuffle in-place.
 *
 * @doc {heading: 'Util', namespace: 'util'}
 */
declare function shuffle(array: any[] | Uint32Array | Int32Array | Float32Array): void;
/**
 * Shuffles two arrays in-place the same way using Fisher-Yates algorithm.
 *
 * ```js
 * const a = [1,2,3,4,5];
 * const b = [11,22,33,44,55];
 * tf.util.shuffleCombo(a, b);
 * console.log(a, b);
 * ```
 *
 * @param array The first array to shuffle in-place.
 * @param array2 The second array to shuffle in-place with the same permutation
 *     as the first array.
 *
 * @doc {heading: 'Util', namespace: 'util'}
 */
declare function shuffleCombo(array: any[] | Uint32Array | Int32Array | Float32Array, array2: any[] | Uint32Array | Int32Array | Float32Array): void;
/** Clamps a value to a specified range. */
declare function clamp(min: number, x: number, max: number): number;
declare function nearestLargerEven(val: number): number;
declare function swap<T>(object: {
    [index: number]: T;
}, left: number, right: number): void;
declare function sum(arr: number[]): number;
/**
 * Returns a sample from a uniform [a, b) distribution.
 *
 * @param a The minimum support (inclusive).
 * @param b The maximum support (exclusive).
 * @return A pseudorandom number on the half-open interval [a,b).
 */
declare function randUniform(a: number, b: number): number;
/** Returns the squared Euclidean distance between two vectors. */
declare function distSquared(a: FlatVector, b: FlatVector): number;
/**
 * Asserts that the expression is true. Otherwise throws an error with the
 * provided message.
 *
 * ```js
 * const x = 2;
 * tf.util.assert(x === 2, 'x is not 2');
 * ```
 *
 * @param expr The expression to assert (as a boolean).
 * @param msg A function that returns the message to report when throwing an
 *     error. We use a function for performance reasons.
 *
 * @doc {heading: 'Util', namespace: 'util'}
 */
declare function assert(expr: boolean, msg: () => string): void;
declare function assertShapesMatch(shapeA: number[], shapeB: number[], errorMessagePrefix?: string): void;
declare function assertNonNull(a: TensorLike): void;
/**
 * Returns the size (number of elements) of the tensor given its shape.
 *
 * ```js
 * const shape = [3, 4, 2];
 * const size = tf.util.sizeFromShape(shape);
 * console.log(size);
 * ```
 *
 * @doc {heading: 'Util', namespace: 'util'}
 */
declare function sizeFromShape(shape: number[]): number;
declare function isScalarShape(shape: number[]): boolean;
declare function arraysEqual(n1: FlatVector, n2: FlatVector): boolean;
declare function isInt(a: number): boolean;
declare function tanh(x: number): number;
declare function sizeToSquarishShape(size: number): [number, number];
/**
 * Creates a new array with randomized indices to a given quantity.
 *
 * ```js
 * const randomTen = tf.util.createShuffledIndices(10);
 * console.log(randomTen);
 * ```
 *
 * @param number Quantity of how many shuffled indices to create.
 *
 * @doc {heading: 'Util', namespace: 'util'}
 */
declare function createShuffledIndices(n: number): Uint32Array;
declare function rightPad(a: string, size: number): string;
declare function repeatedTry(checkFn: () => boolean, delayFn?: (counter: number) => number, maxCounter?: number, scheduleFn?: (functionRef: Function, delay: number) => void): Promise<void>;
/**
 * Given the full size of the array and a shape that may contain -1 as the
 * implicit dimension, returns the inferred shape where -1 is replaced.
 * E.g. For shape=[2, -1, 3] and size=24, it will return [2, 4, 3].
 *
 * @param shape The shape, which may contain -1 in some dimension.
 * @param size The full size (number of elements) of the array.
 * @return The inferred shape where -1 is replaced with the inferred size.
 */
declare function inferFromImplicitShape(shape: number[], size: number): number[];
declare function parseAxisParam(axis: number | number[], shape: number[]): number[];
/** Reduces the shape by removing all dimensions of shape 1. */
declare function squeezeShape(shape: number[], axis?: number[]): {
    newShape: number[];
    keptDims: number[];
};
declare function getTypedArrayFromDType<D extends NumericDataType>(dtype: D, size: number): DataTypeMap[D];
declare function getArrayFromDType<D extends DataType>(dtype: D, size: number): DataTypeMap[D];
declare function checkConversionForErrors<D extends DataType>(vals: DataTypeMap[D] | number[], dtype: D): void;
/** Returns true if the dtype is valid. */
declare function isValidDtype(dtype: DataType): boolean;
/**
 * Returns true if the new type can't encode the old type without loss of
 * precision.
 */
declare function hasEncodingLoss(oldType: DataType, newType: DataType): boolean;
declare function bytesPerElement(dtype: DataType): number;
/**
 * Returns the approximate number of bytes allocated in the string array - 2
 * bytes per character. Computing the exact bytes for a native string in JS
 * is not possible since it depends on the encoding of the html page that
 * serves the website.
 */
declare function bytesFromStringArray(arr: Uint8Array[]): number;
/** Returns true if the value is a string. */
declare function isString(value: {}): value is string;
declare function isBoolean(value: {}): boolean;
declare function isNumber(value: {}): boolean;
declare function inferDtype(values: TensorLike | WebGLData | WebGPUData): DataType;
declare function isFunction(f: Function): boolean;
declare function nearestDivisor(size: number, start: number): number;
declare function computeStrides(shape: number[]): number[];
declare function toNestedArray(shape: number[], a: TypedArray, isComplex?: boolean): number | any[];
declare function convertBackendValuesAndArrayBuffer(data: BackendValues | ArrayBuffer, dtype: DataType): Float32Array | Int32Array | Uint8Array | Uint8Array[];
declare function makeOnesTypedArray<D extends DataType>(size: number, dtype: D): DataTypeMap[D];
declare function makeZerosTypedArray<D extends DataType>(size: number, dtype: D): DataTypeMap[D];
/**
 * Make nested `TypedArray` filled with zeros.
 * @param shape The shape information for the nested array.
 * @param dtype dtype of the array element.
 */
declare function makeZerosNestedTypedArray<D extends DataType>(shape: number[], dtype: D): number | any[];
declare function assertNonNegativeIntegerDimensions(shape: number[]): void;
/**
 * Computes flat index for a given location (multidimentionsal index) in a
 * Tensor/multidimensional array.
 *
 * @param locs Location in the tensor.
 * @param rank Rank of the tensor.
 * @param strides Tensor strides.
 */
declare function locToIndex(locs: number[], rank: number, strides: number[]): number;
/**
 * Computes the location (multidimensional index) in a
 * tensor/multidimentional array for a given flat index.
 *
 * @param index Index in flat array.
 * @param rank Rank of tensor.
 * @param strides Strides of tensor.
 */
declare function indexToLoc(index: number, rank: number, strides: number[]): number[];
/**
 * This method asserts whether an object is a Promise instance.
 * @param object
 */
declare function isPromise(object: any): object is Promise<unknown>;

/// <amd-module name="@tensorflow/tfjs-core/dist/util_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/variable" />
/**
 * Creates a new variable with the provided initial value.
 * ```js
 * const x = tf.variable(tf.tensor([1, 2, 3]));
 * x.assign(tf.tensor([4, 5, 6]));
 *
 * x.print();
 * ```
 *
 * @param initialValue Initial value for the tensor.
 * @param trainable If true, optimizers are allowed to update it.
 * @param name Name of the variable. Defaults to a unique id.
 * @param dtype If set, initialValue will be converted to the given type.
 *
 * @doc {heading: 'Tensors', subheading: 'Creation'}
 */
declare function variable<R extends Rank>(initialValue: Tensor<R>, trainable?: boolean, name?: string, dtype?: DataType): Variable<R>;

/// <amd-module name="@tensorflow/tfjs-layers/dist/variables" />
/**
 * A `tf.layers.LayerVariable` is similar to a `tf.Tensor` in that it has a
 * dtype and shape, but its value is mutable.  The value is itself represented
 * as a`tf.Tensor`, and can be read with the `read()` method and updated with
 * the `write()` method.
 */
declare class LayerVariable {
    readonly dtype: DataType;
    readonly shape: Shape;
    readonly id: number;
    readonly name: string;
    readonly originalName: string;
    private trainable_;
    protected readonly val: tfc.Variable;
    readonly constraint: Constraint;
    /**
     * Construct Variable from a `tf.Tensor`.
     *
     * If not explicitly named, the Variable will be given a name with the
     * prefix 'Variable'. Variable names are unique. In the case of name
     * collision, suffixies '_<num>' will be added to the name.
     *
     * @param val Initial value of the Variable.
     * @param name Name of the variable. If `null` or `undefined` is provided, it
     *   will default a name with the prefix 'Variable'.
     * @param constraint Optional, projection function to be applied to the
     * variable after optimize updates
     * @throws ValueError if `name` is `null` or `undefined`.
     */
    constructor(val: Tensor, dtype?: DataType, name?: string, trainable?: boolean, constraint?: Constraint);
    /**
     * Get a snapshot of the Variable's value.
     *
     * The returned value is a snapshot of the Variable's value at the time of
     * the invocation. Future mutations in the value of the tensor will only
     * be reflected by future calls to this method.
     */
    read(): Tensor;
    /**
     * Update the value of the Variable.
     *
     * @param newVal: The new value to update to. Must be consistent with the
     *   dtype and shape of the Variable.
     * @return This Variable.
     */
    write(newVal: Tensor): this;
    /**
     * Dispose this LayersVariable instance from memory.
     */
    dispose(): void;
    protected assertNotDisposed(): void;
    get trainable(): boolean;
    set trainable(trainable: boolean);
}
/**
 * Create a Variable.
 * @param x The initial value of the `Variable`.
 * @param dtype optional, the type of the variable.
 * @param name optional, the name of the variable, default provided by
 * Variable.
 * @param constraint optional, a constraint to be applied after every update.
 * @return The newly instantiated `Variable`.
 */
declare function variable(x: Tensor, dtype?: DataType, name?: string, constraint?: Constraint): LayerVariable;
/**
 * Instantiates an all-zeros Variable and returns it.
 *
 * @param shape Shape of the tensor.
 * @param dtype DType of the tensor.
 * @param name Name of the tensor.
 * @return An all-zero Variable.
 */
declare function zerosVariable(shape: Shape, dtype?: DataType, name?: string): LayerVariable;
/**
 * Instantiates an all-zeros tensor of the same shape as another tensor.
 *
 * @param x The other tensor.
 * @param dtype DType of the tensor.
 * @param name Name of the tensor.
 * @return A newly instantiated Variable.
 */
declare function zerosLike(x: Tensor, dtype?: DataType, name?: string): LayerVariable;
/**
 * Instantiates an all-ones tensor and returns it.
 *
 * @param shape Shape of the tensor.
 * @param dtype DType of the tensor.
 * @param name Name of the tensor.
 * @return An all-ones Variable.
 */
declare function onesVariable(shape: Shape, dtype?: DataType, name?: string): LayerVariable;
/**
 * Instantiates an all-ones tensor of the same shape as another tensor.
 *
 * @param x The other tensor.
 * @param dtype DType of the tensor.
 * @param name Name of the tensor.
 * @return A newly instantiated Variable.
 */
declare function onesLike(x: Tensor, dtype?: DataType, name?: string): LayerVariable;
/**
 * Instantiate an identity matrix and returns it, as a Variable
 *
 * @param size Number of rows/columns.
 * @param dtype Data type of returned Variable.
 * @param name Name of returned Variable.
 * @return A Variable, an identity matrix.
 */
declare function eyeVariable(size: number, dtype?: DataType, name?: string): LayerVariable;
/**
 * Get a Variable with uniform distribution of values.
 * @param shape Shape of the tensor.
 * @param minval Lower bound of the uniform distribution.
 * @param maxval Upper bound of the uniform distribution.
 * @param dtype
 * @param seed
 * @param name Optional name.
 * @return The uniform-random Variable.
 */
declare function randomUniformVariable(shape: Shape, minval: number, maxval: number, dtype?: DataType, seed?: number, name?: string): LayerVariable;
/**
 * Get a Variable with truncated-normal distribution of values.
 * @param shape Shape of the tensor.
 * @param mean mean value of the normal distribution.
 * @param stddev standard deviation of the normal distribution.
 * @param dtype
 * @param seed
 * @param name Optional name.
 * @return The truncated-normal-random Variable.
 */
declare function truncatedNormalVariable(shape: Shape, mean?: number, stddev?: number, dtype?: DataType, seed?: number, name?: string): LayerVariable;
/**
 * Get a Variable with normal distribution of values.
 * @param shape Shape of the tensor.
 * @param mean mean value of the normal distribution.
 * @param stddev standard deviation of the normal distribution.
 * @param dtype
 * @param seed
 * @param name Optional name.
 * @return The truncated-normal-random Variable.
 */
declare function randomNormalVariable(shape: Shape, mean?: number, stddev?: number, dtype?: DataType, seed?: number, name?: string): LayerVariable;
/**
 * Update the value of a Variable.
 * @param x The Variable to be updated.
 * @param xNew The new value to update to.
 * @return The Variable updated.
 */
declare function update(x: LayerVariable, xNew: Tensor): LayerVariable;
/**
 * Update the value of a Variable by adding an increment.
 * @param x The Variable to be updated.
 * @param increment The incrment to add to `x`.
 * @return The Variable updated.
 */
declare function updateAdd(x: LayerVariable, increment: Tensor): LayerVariable;
/**
 * Update the value of a Variable by subtracting a decrement.
 * @param x The Variable to be updated.
 * @param decrement The decrement to subtract from `x`.
 * @return The Variable updated.
 */
declare function updateSub(x: LayerVariable, decrement: Tensor): LayerVariable;
/**
 * Get the values of an array of Variables.
 *
 * @param tensors An `Array` of `Variable`s to get the values of.
 * @return The values of the inputs, as an `Array` of`tf.Tensor`s.
 */
declare function batchGetValue(xs: LayerVariable[]): Tensor[];
/**
 * Update the value of multiple Variables at once.
 *
 * @param variablesAndValues An `Array`, each element is of type
 *   [Variable, Tensor]. The first item is the
 *   `Variable` of which the value is to be updated. The second item
 *   carries the new value.
 */
declare function batchSetValue(variablesAndValues: Array<[LayerVariable, Tensor]>): void;
/**
 * Returns the gradients of `variables` w.r.t. the return value of `lossFn`.
 * @param lossFn A function which returns a Scalar to be used as the function
 *   value (i.e., numerator) for differentiation.
 * @param variables List of variables to be used as the independent variables
 *   (i.e., denominator) for differentiation.
 * @returns An Array of gradients tensors.
 */
declare function gradients(lossFn: () => tfc.Scalar, variables: LayerVariable[]): Tensor[];

/// <amd-module name="@tensorflow/tfjs-core/dist/variable_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-layers/dist/utils/variable_utils" />
/**
 * Count the elements in an Array of LayerVariables.
 *
 * @param weights: The LayerVariables of which the constituent numbers are to
 *   be counted.
 * @returns A count of the elements in all the LayerVariables
 */
declare function countParamsInWeights(weights: LayerVariable[]): number;

/// <amd-module name="@tensorflow/tfjs-layers/dist/version" />
declare const version = "4.2.0";
{ version };

/// <amd-module name="@tensorflow/tfjs-core/dist/version_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-data/dist/iterators/webcam_iterator" />
/**
 * Provide a stream of image tensors from webcam video stream. Only works in
 * browser environment.
 */
declare class WebcamIterator extends LazyIterator<Tensor3D> {
    protected readonly webcamVideoElement: HTMLVideoElement;
    protected readonly webcamConfig: WebcamConfig;
    private isClosed;
    private stream;
    private resize;
    private cropSize;
    private cropBox;
    private cropBoxInd;
    private constructor();
    summary(): string;
    static create(webcamVideoElement?: HTMLVideoElement, webcamConfig?: WebcamConfig): Promise<WebcamIterator>;
    start(): Promise<void>;
    next(): Promise<IteratorResult<Tensor3D>>;
    private needToResize;
    cropAndResizeFrame(img: Tensor3D): Tensor3D;
    capture(): Promise<Tensor3D>;
    stop(): void;
    toArray(): Promise<Tensor3D[]>;
}

/// <amd-module name="@tensorflow/tfjs-core/dist/io/weights_loader" />
/**
 * Reads binary weights data from a number of URLs.
 *
 * @param fetchURLs URLs to send the HTTP requests at, using `fetch` calls.
 * @param requestOptions RequestInit (options) for the HTTP requests.
 * @param fetchFunc Optional overriding value for the `window.fetch` function.
 * @param onProgress Optional, progress callback function, fired periodically
 *   before the load is completed.
 * @returns A `Promise` of an Array of `ArrayBuffer`. The Array has the same
 *   length as `fetchURLs`.
 */
declare function loadWeightsAsArrayBuffer(fetchURLs: string[], loadOptions?: LoadOptions): Promise<ArrayBuffer[]>;
/**
 * Reads a weights manifest JSON configuration, fetches the weights and
 * returns them as `Tensor`s.
 *
 * @param manifest The weights manifest JSON.
 * @param filePathPrefix The path prefix for filenames given in the manifest.
 *     Defaults to the empty string.
 * @param weightNames The names of the weights to be fetched.
 */
declare function loadWeights(manifest: WeightsManifestConfig, filePathPrefix?: string, weightNames?: string[], requestInit?: RequestInit): Promise<NamedTensorMap>;
/**
 * Creates a function, which reads a weights manifest JSON configuration,
 * fetches the weight files using the specified function and returns them as
 * `Tensor`s.
 *
 * ```js
 * // example for creating a nodejs weight loader, which reads the weight files
 * // from disk using fs.readFileSync
 *
 * import * as fs from 'fs'
 *
 * const fetchWeightsFromDisk = (filePaths: string[]) =>
 *   filePaths.map(filePath => fs.readFileSync(filePath).buffer)
 *
 * const loadWeights = tf.io.weightsLoaderFactory(fetchWeightsFromDisk)
 *
 * const manifest = JSON.parse(
 *   fs.readFileSync('./my_model-weights_manifest').toString()
 * )
 * const weightMap = await loadWeights(manifest, './')
 * ```
 * @param fetchWeightsFunction The function used for fetching the weight files.
 * @returns Weight loading function.
 */
declare function weightsLoaderFactory(fetchWeightsFunction: (fetchUrls: string[]) => Promise<ArrayBuffer[]>): (manifest: WeightsManifestConfig, filePathPrefix?: string, weightNames?: string[]) => Promise<NamedTensorMap>;
/// <amd-module name="@tensorflow/tfjs-core/dist/io/weights_loader_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/where" />
/**
 * Returns the elements, either `a` or `b` depending on the `condition`.
 *
 * If the condition is true, select from `a`, otherwise select from `b`.
 *
 * ```js
 * const cond = tf.tensor1d([false, false, true], 'bool');
 * const a = tf.tensor1d([1 , 2, 3]);
 * const b = tf.tensor1d([-1, -2, -3]);
 *
 * a.where(cond, b).print();
 * ```
 *
 * @param condition The input condition. Must be of dtype bool.
 * @param a If `condition` is rank 1, `a` may have a higher rank but
 *     its first dimension must match the size of `condition`.
 * @param b A tensor with the same dtype as `a` and with shape that is
 *     compatible with `a`.
 * @return A tensor with same dtype as `a` and `b`, and shape that is
 *     broadcastable from `a` and `b`.
 *
 * @doc {heading: 'Operations', subheading: 'Logical'}
 */
declare function where_<T extends Tensor>(condition: Tensor | TensorLike, a: T | TensorLike, b: T | TensorLike): T;
declare const where: typeof where_;
{ };
/// <amd-module name="@tensorflow/tfjs-core/dist/ops/where_async" />
/**
 * Returns the coordinates of true elements of condition.
 *
 * The coordinates are returned in a 2-D tensor where the first dimension (rows)
 * represents the number of true elements, and the second dimension (columns)
 * represents the coordinates of the true elements. Keep in mind, the shape of
 * the output tensor can vary depending on how many true values there are in
 * input. Indices are output in row-major order. The resulting tensor has the
 * shape `[numTrueElems, condition.rank]`.
 *
 * This is analogous to calling the python `tf.where(cond)` without an x or y.
 *
 * ```js
 * const cond = tf.tensor1d([false, false, true], 'bool');
 * const result = await tf.whereAsync(cond);
 * result.print();
 * ```
 *
 * @doc {heading: 'Operations', subheading: 'Logical'}
 */
declare function whereAsync_(condition: Tensor | TensorLike): Promise<Tensor2D>;
declare const whereAsync: typeof whereAsync_;
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/where_async_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/backends/where_impl" />
declare function whereImpl(condShape: number[], condVals: TypedArray): Tensor2D;
/// <amd-module name="@tensorflow/tfjs-core/dist/ops/where_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-layers/dist/layers/wrappers" />
declare interface WrapperLayerArgs extends LayerArgs {
    /**
     * The layer to be wrapped.
     */
    layer: Layer;
}
/**
 * Abstract wrapper base class.
 *
 * Wrappers take another layer and augment it in various ways.
 * Do not use this class as a layer, it is only an abstract base class.
 * Two usable wrappers are the `TimeDistributed` and `Bidirectional` wrappers.
 */
declare abstract class Wrapper extends Layer {
    readonly layer: Layer;
    constructor(args: WrapperLayerArgs);
    build(inputShape: Shape | Shape[]): void;
    get trainable(): boolean;
    set trainable(value: boolean);
    get trainableWeights(): LayerVariable[];
    get nonTrainableWeights(): LayerVariable[];
    get updates(): Tensor[];
    get losses(): RegularizerFn[];
    getWeights(): Tensor[];
    setWeights(weights: Tensor[]): void;
    getConfig(): serialization.ConfigDict;
    setFastWeightInitDuringBuild(value: boolean): void;
    /** @nocollapse */
    static fromConfig<T extends serialization.Serializable>(cls: serialization.SerializableConstructor<T>, config: serialization.ConfigDict, customObjects?: serialization.ConfigDict): T;
}
declare class TimeDistributed extends Wrapper {
    /** @nocollapse */
    static className: string;
    constructor(args: WrapperLayerArgs);
    build(inputShape: Shape | Shape[]): void;
    computeOutputShape(inputShape: Shape | Shape[]): Shape | Shape[];
    call(inputs: Tensor | Tensor[], kwargs: Kwargs): Tensor | Tensor[];
}
declare function checkBidirectionalMergeMode(value?: string): void;
declare interface BidirectionalLayerArgs extends WrapperLayerArgs {
    /**
     * The instance of an `RNN` layer to be wrapped.
     */
    layer: RNN;
    /**
     * Mode by which outputs of the forward and backward RNNs are
     * combined. If `null` or `undefined`, the output will not be
     * combined, they will be returned as an `Array`.
     *
     * If `undefined` (i.e., not provided), defaults to `'concat'`.
     */
    mergeMode?: BidirectionalMergeMode;
}
declare class Bidirectional extends Wrapper {
    /** @nocollapse */
    static className: string;
    mergeMode: BidirectionalMergeMode;
    private forwardLayer;
    private backwardLayer;
    private returnSequences;
    private returnState;
    private numConstants?;
    private _trainable;
    constructor(args: BidirectionalLayerArgs);
    get trainable(): boolean;
    set trainable(value: boolean);
    getWeights(): Tensor[];
    setWeights(weights: Tensor[]): void;
    computeOutputShape(inputShape: Shape | Shape[]): Shape | Shape[];
    apply(inputs: Tensor | Tensor[] | SymbolicTensor | SymbolicTensor[], kwargs?: Kwargs): Tensor | Tensor[] | SymbolicTensor | SymbolicTensor[];
    call(inputs: Tensor | Tensor[], kwargs: Kwargs): Tensor | Tensor[];
    resetStates(states?: Tensor | Tensor[]): void;
    build(inputShape: Shape | Shape[]): void;
    computeMask(inputs: Tensor | Tensor[], mask?: Tensor | Tensor[]): Tensor | Tensor[];
    get trainableWeights(): LayerVariable[];
    get nonTrainableWeights(): LayerVariable[];
    setFastWeightInitDuringBuild(value: boolean): void;
    getConfig(): serialization.ConfigDict;
    /** @nocollapse */
    static fromConfig<T extends serialization.Serializable>(cls: serialization.SerializableConstructor<T>, config: serialization.ConfigDict): T;
}

/// <amd-module name="@tensorflow/tfjs-layers/dist/keras_format/layers/wrappers_serialization" />
declare type TimeDistributedLayerSerialization = BaseLayerSerialization<'TimeDistributed', TimeDistributedLayerConfig>;
interface TimeDistributedLayerConfig extends LayerConfig {
    layer: LayerSerialization;
}
declare type BidirectionalLayerSerialization = BaseLayerSerialization<'Bidirectional', BidirectionalLayerConfig>;
interface BidirectionalLayerConfig extends LayerConfig {
    layer: RecurrentLayerSerialization;
    merge_mode?: BidirectionalMergeMode;
}
declare type WrapperLayerSerialization = TimeDistributedLayerSerialization | BidirectionalLayerSerialization;
declare type WrapperLayerClassName = WrapperLayerSerialization['class_name'];
/**
 * A string array of valid WrapperLayer class names.
 *
 * This is guaranteed to match the `WrapperLayerClassName` union type.
 */
declare const wrapperLayerClassNames: WrapperLayerClassName[];

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/zeros" />
/**
 * Creates a `tf.Tensor` with all elements set to 0.
 *
 * ```js
 * tf.zeros([2, 2]).print();
 * ```
 *
 * @param shape An array of integers defining the output tensor shape.
 * @param dtype The type of an element in the resulting tensor. Can
 *     be 'float32', 'int32' or 'bool'. Defaults to 'float'.
 *
 * @doc {heading: 'Tensors', subheading: 'Creation'}
 */
declare function zeros<R extends Rank>(shape: ShapeMap[R], dtype?: DataType): Tensor<R>;

/// <amd-module name="@tensorflow/tfjs-core/dist/gradients/ZerosLike_grad" />
declare const zerosLikeGradConfig: GradConfig;

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/zeros_like" />
/**
 * Creates a `tf.Tensor` with all elements set to 0 with the same shape as the
 * given tensor.
 *
 * ```js
 * const x = tf.tensor([1, 2]);
 * tf.zerosLike(x).print();
 * ```
 *
 * @param x The tensor of required shape.
 *
 * @doc {heading: 'Tensors', subheading: 'Creation'}
 */
declare function zerosLike_<T extends Tensor>(x: T | TensorLike): T;
declare const zerosLike: typeof zerosLike_;
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/zeros_like_test" />
{ };

/// <amd-module name="@tensorflow/tfjs-core/dist/ops/zeros_test" />
{ };
}