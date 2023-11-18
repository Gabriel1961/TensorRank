import { Component, For, Show } from "solid-js";
import styles from "./TaskTest.module.scss"
import Spinner from "../../../components/Spinner";
import { FaRegularCircleCheck } from 'solid-icons/fa'
import { FaRegularCircleRight } from 'solid-icons/fa'
export interface TestEval {
  done: boolean;
  accuracy: number;
}

interface TaskTestProps {
  tests: TestEval[]
}

const TaskTest: Component<TaskTestProps> = (props) => {
  const getFinalAccuracy = () => {
    let acc = 0;
    for(let i=0;i<props.tests.length;i++){
      const test = props.tests[i]
      if (!test.done)
        return null;
      acc += test.accuracy;
    }

    acc /= props.tests.length;
    return acc
  }
  return <div class={styles.container}>
    <For each={props.tests}>
      {test => <div class={styles.testContainer}>
        <Show when={!test.done} fallback={
          <>
            <FaRegularCircleCheck size={40} color={"greenyellow"} />
            <h3 style={{ "margin-left": "20px" }}>
              Test Accuracy:  {(test.accuracy * 100).toFixed(1)}%
            </h3>
          </>
        }>
          <Spinner />
          <h3 style={{ "margin-left": "20px" }}>Running Test...</h3>
        </Show>
      </div>}
    </For>
    <Show when={getFinalAccuracy()}>
      <div class={styles.testContainer}>
        <FaRegularCircleRight size={40} color={"greenyellow"} />
        <h3 style={{ "margin-left": "20px" }}>Final Accuracy: {(getFinalAccuracy()!*100).toFixed(1)}%</h3>
      </div>
    </Show>
  </div>
}

export default TaskTest;