import { Component, For, Show, createEffect, createSignal } from "solid-js"
import TaskPanelDisplay from "./Components/TaskPanelDisplay";
import { Task } from "../../datamodel";
import styles from "./TasksPage.module.scss"
import { fetchMoreTasks } from "./TasksPageFetcher";
import Spinner from "../../components/Spinner";

const TasksPage: Component = () => {
  const [getTasks, setTasks] = createSignal<Task[]>([])
  const [getIsLoading, setIsLoading] = createSignal<boolean>(false)

  const loadMoreTasks = async (tasks: Task[]) => {
    setIsLoading(true);
    const newTasks = await fetchMoreTasks(tasks)
    setTasks(newTasks)
    setIsLoading(false)
  }
  createEffect(() => {
    loadMoreTasks([])
  })

  return <div class={styles.container}>
    <div class={styles.filterPanel}>
      <h2>Filter</h2>
      <div style={{ margin: "10px", "margin-bottom": "0px", }}>
        <input type="checkbox" checked />
        <label style={{ "margin-left": "10px" }} >Classification</label>
      </div>
      <div style={{ margin: "10px", "margin-bottom": "0px", }}>
        <input type="checkbox" checked/>
        <label style={{ "margin-left": "10px" }} >Regression</label>
      </div>
      <div style={{ margin: "10px", "margin-bottom": "0px", }}>
        <input type="checkbox" checked/>
        <label style={{ "margin-left": "10px" }} >Easy</label>
      </div>

      <div style={{ margin: "10px", "margin-bottom": "0px", }}>
        <input type="checkbox" checked />
        <label style={{ "margin-left": "10px" }} >Medium</label>
      </div>

      <div style={{ margin: "10px", "margin-bottom": "0px", }}>
        <input type="checkbox" />
        <label style={{ "margin-left": "10px" }} >Hard</label>
      </div>
    </div>
    <div class={styles.containerList}>
      <For each={getTasks()}>
        {(task) => <TaskPanelDisplay task={task} />}
      </For>
      <Show when={getIsLoading() === false} fallback={<Spinner />}>
        <button class={styles.loadMore} onClick={() => loadMoreTasks(getTasks())}>Load More</button>
      </Show>
    </div>
  </div>
}

export default TasksPage;