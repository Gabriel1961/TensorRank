import { Component, For, Show, createSignal } from "solid-js";
import { DatasetData } from "../../../datamodel";
import styles from "./TaskTable.module.scss"
interface TaskTableProps {
  dataset: DatasetData | null
}

export const TaskTable: Component<TaskTableProps> = (props) => {
  const [getOpen, setOpen] = createSignal<boolean>(false);
  return <div class={styles.container}>
    <button class={styles.button} onClick={() => setOpen(!getOpen())}>View Dataset Head</button>
    <Show when={getOpen() && props.dataset}>
      <table>
        <thead>
          <tr>
            <For each={props.dataset!.headers}>
              {(head) => <th>{head}</th>}
            </For>
          </tr>
        </thead>
        <tbody>
          <For each={props.dataset?.dataHead}>
            {(row) => <tr>
              <For each={row}>
                {val => <td>{val}</td>}
              </For>
            </tr>}
          </For>
          <tr>
            <For each={props.dataset!.headers}>
              {() => <th>{"..."}</th>}
            </For>
          </tr>
        </tbody>
      </table>
    </Show>
  </div>
}