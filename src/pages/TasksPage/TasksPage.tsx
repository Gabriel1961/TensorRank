import { Component, For, Show, createEffect, createSignal } from "solid-js"
import TaskPanelDisplay from "./Components/TaskPanelDisplay";
import { Task } from "../../datamodel";
import styles from "./TasksPage.module.scss"
import { fetchMoreTasks } from "./TasksPageFetcher";
import Spinner from "../../components/Spinner";

const TasksPage: Component = () => {
  const [getTasks, setTasks] = createSignal<Task[]>([])
  const [getIsLoading, setIsLoading] = createSignal<boolean>(false)
  
  const loadMoreTasks = async (tasks:Task[]) => {
    setIsLoading(true);
    const newTasks = await fetchMoreTasks(tasks)
    setTasks(newTasks)
    setIsLoading(false)
  }
  createEffect(() => {
    loadMoreTasks([])
  })

  return <div class={styles.container}>
    <For each={getTasks()}>
      {(task) => <TaskPanelDisplay task={task} />}
    </For>
    <Show when={getIsLoading() === false} fallback={<Spinner />}>
      <button onClick={()=>loadMoreTasks(getTasks())}>Load More</button>
    </Show>
  </div>
}

export default TasksPage;