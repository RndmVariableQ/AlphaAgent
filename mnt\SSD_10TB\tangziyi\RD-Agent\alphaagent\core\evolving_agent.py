    def multistep_evolve(
        self,
        evo: EvolvableSubjects,  # 可进化主体对象
        eva: Evaluator | Feedback,  # 评估器或反馈对象
        filter_final_evo: bool = False,  # 是否在最后过滤进化结果
    ) -> EvolvableSubjects:
        """
        多步进化方法，实现了完整的进化-评估循环流程
        
        参数:
            evo: 待进化的主体对象
            eva: 用于评估进化结果的评估器或直接的反馈对象
            filter_final_evo: 是否在进化结束后根据最后的反馈过滤结果
            
        返回:
            进化后的主体对象
        """
        # 在进度条中执行最大循环次数的迭代
        for _ in tqdm(range(self.max_loop), "Debugging"):
            # 1. 知识自进化阶段
            # 如果启用了知识自生成且RAG模型存在，则生成新的知识
            if self.knowledge_self_gen and self.rag is not None:
                self.rag.generate_knowledge(self.evolving_trace)

            # 2. RAG查询阶段
            # 初始化查询知识为空
            queried_knowledge = None
            # 如果启用了知识查询且RAG模型存在，则执行查询
            if self.with_knowledge and self.rag is not None:
                queried_knowledge = self.rag.query(evo, self.evolving_trace)

            # 3. 进化阶段
            # 使用进化策略对主体进行进化，传入进化追踪记录和查询到的知识
            evo = self.evolving_strategy.evolve(
                evo=evo,
                evolving_trace=self.evolving_trace,
                queried_knowledge=queried_knowledge,
            )
            # 记录进化后的代码工作空间信息
            logger.log_object(evo.sub_workspace_list, tag="evolving code")
            for sw in evo.sub_workspace_list:
                logger.info(f"evolving code workspace: {sw}")

            # 4. 打包进化结果
            # 创建进化步骤对象，记录当前进化状态和查询到的知识
            es = EvoStep(evo, queried_knowledge)

            # 5. 评估阶段
            # 如果启用了反馈功能，则评估当前进化结果
            if self.with_feedback:
                es.feedback = (
                    eva if isinstance(eva, Feedback)
                    else eva.evaluate(evo, queried_knowledge=queried_knowledge)
                )
                # 记录评估反馈
                logger.log_object(es.feedback, tag="evolving feedback")

            # 6. 更新进化追踪记录
            # 将当前进化步骤添加到追踪列表中
            self.evolving_trace.append(es)

        # 进化循环结束后的处理
        # 如果启用了反馈且需要过滤，则根据最后一次反馈过滤进化结果
        if self.with_feedback and filter_final_evo:
            evo = self.filter_evolvable_subjects_by_feedback(evo, self.evolving_trace[-1].feedback)
            
        return evo 