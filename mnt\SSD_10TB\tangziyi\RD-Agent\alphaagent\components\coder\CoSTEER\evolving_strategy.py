def evolve(
    self,
    *,
    evo: EvolvingItem,
    queried_knowledge: CoSTEERQueriedKnowledge | None = None,  # 保留参数但不使用
    **kwargs,
) -> EvolvingItem:
    # 1.找出需要evolve的task
    to_be_finished_task_index = []
    for index, target_task in enumerate(evo.sub_tasks):
        # target_task_desc = target_task.get_task_information()
        # if target_task_desc in queried_knowledge.success_task_to_knowledge_dict:
        #     evo.sub_workspace_list[index] = queried_knowledge.success_task_to_knowledge_dict[
        #         target_task_desc
        #     ].implementation
        # elif (
        #     target_task_desc not in queried_knowledge.success_task_to_knowledge_dict
        #     and target_task_desc not in queried_knowledge.failed_task_info_set
        # ):
        #     to_be_finished_task_index.append(index)
        to_be_finished_task_index.append(index)  # 现在直接添加所有任务

    # 2. 选择selection方法
    if self.settings.select_threshold < len(to_be_finished_task_index):
        # Select a fixed number of factors if the total exceeds the threshold
        to_be_finished_task_index = self.select_one_round_tasks(
            to_be_finished_task_index, 
            evo, 
            self.settings.select_threshold, 
            queried_knowledge,  # 这个参数会在select_one_round_tasks中被忽略
            self.scen
        )

    result = multiprocessing_wrapper(
        [
            (self.implement_one_task, (evo.sub_tasks[target_index], None))  # 将queried_knowledge改为None
            for target_index in to_be_finished_task_index
        ],
        n=RD_AGENT_SETTINGS.multi_proc_n,
    )
    code_list = [None for _ in range(len(evo.sub_tasks))]
    for index, target_index in enumerate(to_be_finished_task_index):
        code_list[target_index] = result[index]

    evo = self.assign_code_list_to_evo(code_list, evo)
    evo.corresponding_selection = to_be_finished_task_index

    return evo 